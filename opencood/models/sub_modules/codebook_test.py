import copy

Eps = 1e-6

import math
from typing import Callable, Dict, List, Tuple, Union
# from entropyCoder import EntropyCoder

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from opencood.utils.codebook_utils import LowerBound, gumbelSoftmax, CodeSize


class BaseQuantizer(nn.Module):
    def __init__(self, m: int, k: List[int]):
        super().__init__()
        # self._entropyCoder = EntropyCoder(m, k)
        self._m = m
        self._k = k

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def Codebooks(self) -> List[torch.Tensor]:
        raise NotImplementedError

    @property
    def CDFs(self):
        return self._entropyCoder.CDFs

    def reAssignCodebook(self) -> torch.Tensor:
        raise NotImplementedError

    def syncCodebook(self):
        raise NotImplementedError

    @property
    def NormalizedFreq(self):
        return self._entropyCoder.NormalizedFreq

    def compress(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[bytes]], List[CodeSize]]:
        codes = self.encode(x)

        # List of binary, len = n, len(binaries[0]) = level
        binaries, codeSize = self._entropyCoder.compress(codes)
        return codes, binaries, codeSize

    def _validateCode(self, refCodes: List[torch.Tensor], decompressed: List[torch.Tensor]):
        for code, restored in zip(refCodes, decompressed):
            if torch.any(code != restored):
                raise RuntimeError("Got wrong decompressed result from entropy coder.")

    def decompress(self, binaries: List[List[bytes]], codeSize: List[CodeSize]) -> torch.Tensor:
        decompressed = self._entropyCoder.decompress(binaries, codeSize)
        # self._validateCode(codes, decompressed)
        return self.decode(decompressed)


class _multiCodebookQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter, permutationRate: float = 0.0):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self._scale = math.sqrt(self._k)
        self._temperature = nn.Parameter(torch.ones((self._m, 1)))
        self._bound = LowerBound(Eps)
        self._permutationRate = permutationRate

    def reAssignCodebook(self, freq: torch.Tensor) -> torch.Tensor:
        codebook = self._codebook.clone().detach()
        freq = freq.to(self._codebook.device).clone().detach()
        #       [k, d],        [k]
        for m, (codebookGroup, freqGroup) in enumerate(zip(self._codebook, freq)):
            neverAssignedLoc = freqGroup < Eps
            totalNeverAssigned = int(neverAssignedLoc.sum())
            # More than half are never assigned
            if totalNeverAssigned > self._k // 2:
                mask = torch.zeros((totalNeverAssigned,), device=self._codebook.device)
                maskIdx = torch.randperm(len(mask))[self._k // 2:]
                # Random pick some never assigned loc and drop them.
                mask[maskIdx] = 1.
                freqGroup[neverAssignedLoc] = mask
                # Update
                neverAssignedLoc = freqGroup < Eps
                totalNeverAssigned = int(neverAssignedLoc.sum())
            argIdx = torch.argsort(freqGroup, descending=True)[:(self._k - totalNeverAssigned)]
            mostAssigned = codebookGroup[argIdx]
            selectedIdx = torch.randperm(len(mostAssigned))[:totalNeverAssigned]
            codebook.data[m, neverAssignedLoc] = mostAssigned[selectedIdx]
        # [m, k] bool
        diff = ((codebook - self._codebook) ** 2).sum(-1) > 1e-6
        proportion = diff.flatten()
        self._codebook.data.copy_(codebook)
        return proportion

    def syncCodebook(self):
        # codebook = self._codebook.clone().detach()
        dist.broadcast(self._codebook, 0)

    def encode(self, x: torch.Tensor):
        # [n, m, k]
        distance = self._distance(x)
        # [n, m, k] -> [n, m]
        code = distance.argmin(-1)
        #      [n, m]
        return code

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def _distance(self, x: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        # [n, m, d]
        x = x.reshape(n, self._m, self._d)

        # [n, m, 1]
        x2 = (x ** 2).sum(2, keepdim=True)

        # [m, k, 1, 1]
        c2 = (self._codebook ** 2).sum(-1, keepdim=False)

        # [n, m, d] * [m, k, d] -sum-> [n, m, k]
        inter = torch.einsum("nmd,mkd->nmk", x, self._codebook)
        # [n, m, k]
        distance = x2 + c2 - 2 * inter

        return distance

    def _logit(self, x: torch.Tensor) -> torch.Tensor:
        logit = -1 * self._distance(x)
        return logit / self._scale

    def _permute(self, sample: torch.Tensor) -> torch.Tensor:
        if self._permutationRate < Eps:
            return sample
        # [n, h, w, m]
        needPerm = torch.rand_like(sample[..., 0]) < self._permutationRate
        randomed = F.one_hot(torch.randint(self._k, (needPerm.sum(),), device=sample.device),
                             num_classes=self._k).float()
        sample[needPerm] = randomed
        return sample

    def _sample(self, x: torch.Tensor, temperature: float):
        # [n, m, k] * [m, 1]
        logit = self._logit(x) * self._bound(self._temperature)

        # It causes training unstable
        # leave to future tests.
        # add random mask to pick a different index.
        # [n, m, h, w]
        # needPerm = torch.rand_like(logit[..., 0]) < self._permutationRate * rateScale
        # target will set to zero (one of k) but don't break gradient
        # mask = F.one_hot(torch.randint(self._k, (needPerm.sum(), ), device=logit.device), num_classes=self._k).float() * logit[needPerm]
        # logit[needPerm] -= mask.detach()

        # NOTE: STE: code usage is very low; RelaxedOneHotCat: Doesn't have STE trick
        # So reverse back to F.gumbel_softmax
        # posterior = OneHotCategoricalStraightThrough(logits=logit / temperature)
        # [n, m, k, h, w]
        # sampled = posterior.rsample(())

        sampled = gumbelSoftmax(logit, temperature, True)

        sampled = self._permute(sampled)

        # It causes training unstable
        # leave to future tests.
        # sampled = gumbelArgmaxRandomPerturb(logit, self._permutationRate * rateScale, temperature)
        return sampled, logit

    def forward(self, x: torch.Tensor):
        sample, logit = self._sample(x, 1.0)
        # [n, m, 1]
        code = logit.argmax(-1, keepdim=True)
        # [n, m, k]
        oneHot = torch.zeros_like(logit).scatter_(-1, code, 1)
        # [n, m, k]
        return sample, code[..., 0], oneHot, logit


class _multiCodebookDeQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        # self.register_buffer("_ix", torch.arange(self._m), persistent=False)

    def decode(self, code: torch.Tensor):
        # codes: [n, m]
        n, _ = code.shape
        # [n, m]
        # use codes to index codebook (m, k, d) ==> [n, m, k] -> [n, c]
        ix = self._ix.expand_as(code)
        # [n, m, d]
        indexed = self._codebook[ix, code]
        # [n, c]
        return indexed

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def forward(self, sample: torch.Tensor):
        n, _, _ = sample.shape
        # [n, m, h, w, k, 1], [m, 1, 1, k, d] -sum-> [n, m, h, w, d] -> [n, m, d, h, w] -> [n, c, h, w]
        return torch.einsum("nmk,mkd->nmd", sample, self._codebook).reshape(n, -1)


class _quantizerEncoder(nn.Module):

    def __init__(self, quantizer: _multiCodebookQuantization, dequantizer: _multiCodebookDeQuantization,
                 latentStageEncoder: nn.Module, quantizationHead: nn.Module, latentHead: Union[None, nn.Module]):
        super().__init__()
        self._quantizer = quantizer
        self._dequantizer = dequantizer
        self._latentStageEncoder = latentStageEncoder
        self._quantizationHead = quantizationHead
        self._latentHead = latentHead

    @property
    def Codebook(self):
        return self._quantizer._codebook

    def syncCodebook(self):
        self._quantizer.syncCodebook()

    def reAssignCodebook(self, freq: torch.Tensor) -> torch.Tensor:
        return self._quantizer.reAssignCodebook(freq)

    def encode(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        code = self._quantizer.encode(self._quantizationHead(z))
        if self._latentHead is None:
            return None, code
        z = self._latentHead(z)
        #      �� residual,                         [n, m, h, w]
        return z - self._dequantizer.decode(code), code

    def forward(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        q, code, oneHot, logit = self._quantizer(self._quantizationHead(z))
        if self._latentHead is None:
            return q, None, code, oneHot, logit
        z = self._latentHead(z)
        #         �� residual
        return q, z - self._dequantizer(q), code, oneHot, logit


class _quantizerDecoder(nn.Module):

    def __init__(self, dequantizer: _multiCodebookDeQuantization, dequantizationHead: nn.Module,
                 sideHead: Union[None, nn.Module], restoreHead: nn.Module):
        super().__init__()
        self._dequantizer = dequantizer
        self._dequantizationHead = dequantizationHead
        self._sideHead = sideHead
        self._restoreHead = restoreHead

    #                [n, m, h, w]
    def decode(self, code: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer.decode(code))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)

    def forward(self, q: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer(q))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)


class UMGMQuantizer(BaseQuantizer):
    _components = [
        "latentStageEncoder",
        "quantizationHead",
        "latentHead",
        "dequantizationHead",
        "sideHead",
        "restoreHead"
    ]

    def __init__(self, channel: int, m: int, k: Union[int, List[int]], permutationRate: float,
                 components: Dict[str, Callable[[], nn.Module]]):
        if isinstance(k, int):
            k = [k]
        super().__init__(m, k)

        self.ema = 0.9
        # self._freqEMA = nn.ParameterList(nn.Parameter(torch.ones(m, ki) / ki, requires_grad=False) for ki in k)

        componentFns = [components[key] for key in self._components]
        latentStageEncoderFn, quantizationHeadFn, latentHeadFn, dequantizationHeadFn, sideHeadFn, restoreHeadFn = componentFns

        encoders = list()
        decoders = list()

        for i, ki in enumerate(k):
            latentStageEncoder = latentStageEncoderFn()
            quantizationHead = quantizationHeadFn()
            latentHead = latentHeadFn() if i < len(k) - 1 else None
            dequantizationHead = dequantizationHeadFn()
            sideHead = sideHeadFn() if i < len(k) - 1 else None
            restoreHead = restoreHeadFn()
            # This magic is called SmallInit, from paper
            # "Transformers without Tears: Improving the Normalization of Self-Attention",
            # https://arxiv.org/pdf/1910.05895.pdf
            # I've tried a series of initilizations, but found this works the best.
            codebook = nn.Parameter(
                nn.init.normal_(torch.empty(m, ki, channel // m), std=math.sqrt(2 / (5 * channel / m))))
            quantizer = _multiCodebookQuantization(codebook, permutationRate)
            dequantizer = _multiCodebookDeQuantization(codebook)
            encoders.append(_quantizerEncoder(quantizer, dequantizer, latentStageEncoder, quantizationHead, latentHead))
            decoders.append(_quantizerDecoder(dequantizer, dequantizationHead, sideHead, restoreHead))

        self._encoders: nn.ModuleList[_quantizerEncoder] = nn.ModuleList(encoders)
        self._decoders: nn.ModuleList[_quantizerDecoder] = nn.ModuleList(decoders)

    @property
    def Codebooks(self):
        return list(encoder.Codebook for encoder in self._encoders)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        codes = list()
        for encoder in self._encoders:
            x, code = encoder.encode(x)
            #            [n, m, h, w]
            codes.append(code)
        # lv * [n, m, h, w]
        return codes

    def decode(self, codes: List[torch.Tensor]) -> Union[torch.Tensor, None]:
        formerLevel = None
        for decoder, code in zip(self._decoders[::-1], codes[::-1]):
            formerLevel = decoder.decode(code, formerLevel)
        return formerLevel

    def reAssignCodebook(self) -> torch.Tensor:
        freqs = self.NormalizedFreq
        reassigned: List[torch.Tensor] = list()
        for encoder, freq in zip(self._encoders, freqs):
            # freq: [m, ki]
            reassigned.append(encoder.reAssignCodebook(freq))
        return torch.cat(reassigned).float().mean()

    def syncCodebook(self):
        dist.barrier()
        for encoder in self._encoders:
            encoder.syncCodebook()

    def updateFreq(self, onehot_list):
        for lv, code in enumerate(onehot_list):
            # [n, m, k]
            code = code.sum(0)
            # [m, k]
            normalized = code / code.sum(-1, keepdim=True)
            ema = (1 - self.ema) * normalized + self.ema * self._freqEMA[lv]

            self._freqEMA[lv] = ema

    def normalFreq(self):
        freq = list()
        for freqEMA in self._freqEMA:
            # normalized probs.
            freq.append((freqEMA / freqEMA.sum(-1, keepdim=True)).clone().detach())
        return freq

    def forward(self, x: torch.Tensor):
        x_gt = x.detach()
        quantizeds = list()
        codes = list()
        oneHots = list()
        logits = list()
        #
        # import pdb
        # pdb.set_trace()

        for encoder in self._encoders:
            #          �� residual
            quantized, x, code, oneHot, logit = encoder(x)
            # [n, c, h, w]
            quantizeds.append(quantized)
            # [n, m, h, w]
            codes.append(code)
            # [n, m, h, w, k]
            oneHots.append(oneHot)
            # [n, m, h, w, k]
            logits.append(logit)
        formerLevel = None
        for decoder, quantized in zip(self._decoders[::-1], quantizeds[::-1]):
            # �� restored
            formerLevel = decoder(quantized, formerLevel)
            #print("shape:", formerLevel.shape)

        # update freq in entropy coder

        # remember to calculate prob!!!!!!!!!!!!!
        # self.updateFreq(oneHots)

        code_loss = F.mse_loss(formerLevel, x_gt)

        return formerLevel, codes, logits, code_loss


p_rate = 0.0
seg_num = 2
dict_size = [64, 64, 64]
channel = 64
ChannelCompressor = UMGMQuantizer(channel, seg_num, dict_size, p_rate,
                          {"latentStageEncoder": lambda: nn.Linear(channel, channel), "quantizationHead": lambda: nn.Linear(channel, channel),
                           "latentHead": lambda: nn.Linear(channel, channel), "restoreHead": lambda: nn.Linear(channel, channel),
                           "dequantizationHead": lambda: nn.Linear(channel, channel), "sideHead": lambda: nn.Linear(channel, channel)})