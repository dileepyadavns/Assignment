from socket import * #to import socket
c=socket() #created an instance for socket
c.connect(('localhost',12111)) #to connect to the port
print("connect to server")
while True:
    recData=c.recv(1024).decode()
    print('Server: ',recData)
    data=input("Client : ") #to send message 
    c.send(bytes(data ,'utf-8'))

c.close() #closing the connection
