import requests

# LINEに通知する関数
def line_notify(message):
    line_notify_token = 'RjPVIZw064cOaSknExnBVidycTcfZJiHdCOvB5AMS3k'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token} 
    requests.post(line_notify_api, data=payload, headers=headers)

if __name__ == '__main__':
    message = "Hello world!"
    line_notify(message)