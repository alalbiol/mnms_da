#!/usr/bin/env python
# coding: utf-8
"""
Usage: python tools/notify.py --msg "Hello"
"""
import os
import argparse
import requests
import json
import socket


def parse_args():
    parser = argparse.ArgumentParser(description='Send a slack message to your channel')
    parser.add_argument('--msg', type=str, required=True, help='Message.')
    parser.add_argument('--channel', type=str, default="experiments", help='Channel.')
    aux = parser.parse_args()
    return aux


def slack_message(message, channel, blocks=None):
    # https://keestalkstech.com/2019/10/simple-python-code-to-send-message-to-slack-channel-without-packages/
    if os.environ.get('SLACK_TOKEN') is not None:
        token = os.environ.get('SLACK_TOKEN')
    else:
        assert False, "Please set the environment variable SLACK_TOKEN if you want Slack notifications."
    if channel[0] != "#":
        channel = f"#{channel}"
    message = "[{}] {}".format(socket.gethostname().upper(), message)
    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': token,
        'channel': channel,
        'text': message,
        # https://slackmojis.com/
        'icon_url': "https://emojis.slackmojis.com/emojis/images/1453406830/264/success-kid.png?1453406830",
        'username': "Experiments Bot",
        'blocks': json.dumps(blocks) if blocks else None
    }).json()


args = parse_args()

slack_message(message=args.msg, channel=args.channel)
