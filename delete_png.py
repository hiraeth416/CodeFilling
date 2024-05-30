import os
 
def del_files(path):
    for root , dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"):
                os.remove(os.path.join(root, name))
                print ("Delete File: " + os.path.join(root, name))
 
# test
if __name__ == "__main__":
    path = '/GPFS/public/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL'
    del_files(path)