''' lark api '''
import logging
import requests


def send(content, *whos):
    ''' send lark message '''
    url = 'http://101.35.144.217:7005/'
    json = {
        'receivers': ','.join([str(who) for who in whos]),
        'content': content
    }
    response = requests.post(url, json=json, timeout=3)
    return response


def try_send(content, *whos):
    ''' try send lark message '''
    try:
        return send(content, *whos)
    except Exception as e:
        logging.error(f"send lark message error: {e}")
        return None
