#!/usr/bin/env python
# coding: utf-8

import jenkins
import sys
import argparse
import getpass
import os
parser = argparse.ArgumentParser(__name__)

JENKINS_SERVER_URL = "http://jenkins.nevint.com"
JENKINS_USER_ID = "tortuga"
JENKINS_PWD = "bW3eSuL3Wu3Xew"
server = jenkins.Jenkins(JENKINS_SERVER_URL, username=JENKINS_USER_ID, password=JENKINS_PWD)
parser.add_argument('--recipient_list', type=str, default=None)
parser.add_argument('--subject', type=str, default=None)
parser.add_argument('--content', type=str, default=None)
args = parser.parse_args()
os.environ['http_proxy'] = 'http://10.110.63.27:8080'
os.environ['https_proxy'] = 'http://10.110.63.27:8080'


def get_email_addr():
    name = getpass.getuser()
    if name in ['da.liu', 'liuda']:
        return 'da.liu@nio.com'
    elif name in ['g-nlu', 'nlu']:
        return 'swc_ais_nlu@nextevinc.partner.onmschina.cn'
    return None


def send_email(subject, content, recipient_list=None):
    if recipient_list is None:
        recipient_list = get_email_addr()
    if recipient_list is None:
        print('no recipient, email not send.')
        return

    parameters = {"recipient_list": recipient_list, "subject": subject, "content": content}
    server.build_job('ais-nlu-send-email', parameters)


if __name__ == '__main__':
    try:
        send_email(args.subject, args.content, args.recipient_list)
    except Exception as e:
        res = 'ERROR! trigger jenkins job error.exception=%s.' % (e)
        print(res)
        sys.exit(1)
