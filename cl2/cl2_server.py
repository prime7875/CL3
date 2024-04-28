# server.py
import Pyro4

@Pyro4.expose
class StringConcatenator:
    def concatenate_strings(self, str1, str2):
        return str1 + str2

daemon = Pyro4.Daemon()
uri = daemon.register(StringConcatenator)
print("Ready. Object uri =", uri)
daemon.requestLoop()
