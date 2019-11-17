import classify
import datetime
from json import dumps
from keycloak import jwt
from bottle import route, run,request,response
from keycloak import KeycloakOpenID

def validate(token):
  KEYCLOAK_PUBLIC_KEY = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvgTcVSXga3C6Tam96Q2kmnQ7U55f0hA/56HRQnsY9V0d21QlLkkD+DvIbKGaDVNLp8I+Gqlx4s6u5oDKIA12BbE9jAWz7nblKJUMjESDbpQJVivLAOXPvHQK8tCvxClp8dLEb9cEGXbabO+YIWN3pNn5CL8l3IBbJs1sT76G3jyo0sw5oiSRj0agxIPrtThDPDKhOqG0V+NYtudeE1FhWzt4ajYS4yIHGioVtYsvTfePMRGLRHTMQPjia8kBwnVRDwqMputLGfBg8qnz/xUPiRj5hmEeLuT73zte/CLmfSyqn7IsOrso1zEXgBBLz8XUsp1elGRz+zZtkHqnPot8EQIDAQAB"
  KEYCLOAK_PUBLIC_KEY = f"-----BEGIN PUBLIC KEY-----\n{KEYCLOAK_PUBLIC_KEY}\n-----END PUBLIC KEY-----"
                           

  options = {"verify_signature": True, "exp": True}
  keycloak_openid = KeycloakOpenID(server_url="http://localhost:8080/auth/",
                    client_id="api",
                    realm_name="master",
                    client_secret_key="05e8bfbf-e050-4830-bc8a-d5f4376b13ff")
  try:    
    token_info = keycloak_openid.decode_token(token, KEYCLOAK_PUBLIC_KEY, options=options)
    if(token_info):
      return True
  except jwt.JWTClaimsError:
    return False
  except jwt.ExpiredSignatureError:
    return False
  except jwt.JWSError:
    return False

@route('/classify', method='POST')
def classify():
  json = request.json
  token = request.environ.get('HTTP_AUTHORIZATION','')
  valid = validate(token)
  if valid == False:
     result = {'Error': "Access Denied"}
     response.content_type = 'application/json'
     response.status = 403
     return dumps(result)
  else:
    result = predict.predict(json)
    result = {'classification': result}    
    response.content_type = 'application/json'
    response.status = 200

  return dumps(result)

run(host='localhost', port=8081)