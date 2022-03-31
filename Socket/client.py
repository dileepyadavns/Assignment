from socket import *
c=socket()
c.connect(('localhost',12111))
print("connect to server")
while True:
    recData=c.recv(1024).decode()
    print('Server: ',recData)
    data=input("Client : ")
    c.send(bytes(data ,'utf-8'))

c.close()