from zipa_sys import ZIPA_System

f = open("/home/pi/hostname.txt", "r")
ID = f.read().strip()
f.close()


SERVICE_NAME = "_zipa._tcp.local."

UNCOND_HOST = True

if __name__ == "__main__":
    zipa_sys = ZIPA_System(
        ID, SERVICE_NAME, "/mnt/data/", collection_mode=True, only_locally_store=False)
    zipa_sys.start()
