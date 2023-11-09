import requests
from requests.auth import HTTPBasicAuth


class StateTokenAuth(HTTPBasicAuth):
    def __init__(self, token):
        super().__init__(None, None)
        self.token = token

    def __call__(self, r):
        r.headers['Authorization'] = 'Bearer {}'.format(self.token)
        return r