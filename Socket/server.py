#im porting modules
from socket import *
from time import *

server=socket(AF_INET,SOCK_STREAM)
server.bind(('localhost',12111))

server.listen()
print("Server is listening")
connection,address = server.accept()
print("connected to client")

while True:
    data=input('Server: ') #to send message
    connection.send(bytes(data+ctime(),'utf-8'))

    recData=connection.recv(1024).decode() #decoded the message from client
    print('Client: '+recData) 

connection.close() #closing the connection
