import requests
import bcrypt

# req = requests.post('http://127.0.0.1:5000/user', json = {"name":"Second","pass":"PASSWORD"})
req = requests.get('http://127.0.0.1:5000/users')
# req = requests.post('http://127.0.0.1:5000/auth', json = {"name":"Second","pass":"PASSWORD"}) #52FD17
# req = requests.post('http://127.0.0.1:5000/user/Second', json = {"name":"New name"},headers  = {"auth":"C6993B"})

# passw = b"password"
#
# print(passw)
# print(type(passw))
#
# passw = "password".encode(errors = 'surrogateescape')
#
# print(passw)
# print(type(passw))
#
# hashed = bcrypt.hashpw(passw, bcrypt.gensalt())
#
# print(hashed)
# print(type(hashed))
#
# print()
# s = hashed.decode(errors = 'surrogateescape')
# print(s)
# print(type(s))
# print()
#
# b = s.encode(errors = 'surrogateescape')
#
# print(b)
# print(type(b))
# print()

print(req.text)