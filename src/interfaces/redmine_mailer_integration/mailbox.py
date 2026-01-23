#!/bin/python
import email
import imaplib
import os
import re

from bs4 import BeautifulSoup

from src.utils.runtime_config_loader import load_runtime_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

### DEFINITIONS
# this constant defines an offset into the message description
# which contains the Redmine issue id that a message refers to.
ISSUE_ID_OFFSET = 9


class Mailbox:
    'A class to describe the mailbox usage.'

    def __init__(self, user, password):
        """
        The mailbox (should be a singleton).
        """
        self.mailbox = None
        self.user = user
        self.password = password
        config = load_runtime_config()
        services_config = config.get("services", {}) if isinstance(config, dict) else {}
        self.config = services_config.get("redmine_mailbox", {})

        # make sure to open the mailbox
        if self._verify():
            self.mailbox = self._connect()


    def process_message(self, num, redmine):
        """
        Process a single message, including addition to redmine and removal from inbox
        """
        _, msg_data = self.mailbox.fetch(num, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                # get the basic message parameters (description = body)
                msg = email.message_from_bytes(response_part[1])
                sender, cc, subject, description = self._get_fields(msg)

                # find the issue, if it exists already
                issue_id = self._find_issue_id(description)

                # make sure to deal with attachments correctly
                attachments = self._get_attachments(msg)

                if issue_id > 0:
                    note = f"ISSUE_ID:{issue_id} continued (leave for reference)\n\n"
                    note += f"{subject}: {description}"
                    redmine.reopen_issue(issue_id, note, attachments)
                    self._cleanup_message(num, attachments)
                else:
                    issue_id = redmine.new_issue(sender, cc, subject, description, attachments)
                    if issue_id > 0:
                        self._cleanup_message(num, attachments)
                    else:
                        logger.error(f"issue_id is not well defined: {issue_id}")

        return


    def process_messages(self, redmine):
        """
        Select all messages in the mailbx and process them.
        """
        self.mailbox.select()
        _, data = self.mailbox.search(None, 'ALL')
        logger.info(f"mailbox.process_messages: {len(data[0].split())}")

        for num in data[0].split():
            self.process_message(num, redmine)

        self.mailbox.close()
        self.mailbox.logout()

        return


    def _cleanup_message(self, num, attachments):
        self.mailbox.store(num, '+FLAGS', '\\Deleted')

        # remove temporary attachment copies
        for a in attachments:
            os.system(f"rm /tmp/{a['filename']}")

        return


    def _get_charsets(self, msg):
        charsets = set({})
        for c in msg.get_charsets():
            if c is not None:
                charsets.update([c])

        return charsets


    def _find_issue_id(self, description):
        """
        Select all messages in the mailbx and process them.
        """
        issue_id = 0
        if description is not None:
            index = description.find('ISSUE_ID:')
        else:
            index = -1
            
        if index > 0:
            issue_id = int(description[index + ISSUE_ID_OFFSET:].split()[0])
        logger.info(f"ISSUE_ID: {issue_id}")

        return issue_id


    def _get_attachments(self, msg):
        # finding all attachments in an email
        attachments = []
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            file_name = part.get_filename()

            if bool(file_name):
                logger.info(f"Found attachement: {file_name}")
                file_path = os.path.join('/tmp/', file_name)
                if not os.path.isfile(file_path):
                    with open(file_path, 'wb') as f:
                        f.write(part.get_payload(decode=True))

                    attachments.append({'path': file_path, 'filename': file_name})
                else:
                    logger.error("Could not download attachment (file exists).")
                    return []

        return attachments


    def _get_email_body(self, msg):
        # finding the body in an email message
        body, body_html = None, None

        # walk through the parts of the email to find the text body.    
        if msg.is_multipart():
            for part in msg.walk():

                # if part is multipart, walk through the subparts.            
                if part.is_multipart(): 

                    for subpart in part.walk():
                        if subpart.get_content_type() == 'text/plain' and body == None:
                            # get the subpart payload (i.e the message body)
                            body = subpart.get_payload(decode=True) 
                        elif subpart.get_content_type() == 'html' and body_html == None:
                            body_html = subpart.get_payload(decode=True)

                # part isn't multipart so get the email body
                elif part.get_content_type() == 'text/plain' and body == None:
                    body = part.get_payload(decode=True)

        # if this is not a multi-part message then get the payload (i.e the message body)
        elif msg.get_content_type() == 'text/plain':
            body = msg.get_payload(decode=True)
        elif msg.get_content_type() == 'text/html':
            body_html = msg.get_payload(decode=True)

       # no checking done to match the charset with the correct part. 
        for charset in self._get_charsets(msg):
            try:
                body = body.decode(charset)
            except UnicodeDecodeError:
                self._handle_error("UnicodeDecodeError: encountered.", msg, charset)
            except AttributeError:
                self._handle_error("AttributeError: encountered.", msg, charset)

        return body, body_html


    def _get_fields(self, msg):
        sender, cc, subject = msg['from'], msg['cc'], msg['subject']
        for header in ['subject', 'to', 'cc', 'bcc', 'from']:
            logger.info(f"{header.upper():8s}: {msg[header]}")

        body, body_html = self._get_email_body(msg)
        if body:
            description = body
        else:
            new_body = self._extract_email_body(body_html)
            description = new_body
        description = self._clear_text(description)
        logger.info(f"BODY: {description}")
        return sender, cc, subject, description

    def _clear_text(self,string):
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, '', string)

    def _connect(self):
        """
        Open the mailbox
        """
        logger.info(f"Open mailbox (U:{self.user} P:*********)")
        mailbox = imaplib.IMAP4(host='ppc.mit.edu', port=self.config["imap4_port"], timeout=None)
        mailbox.login(self.user, self.password)

        return mailbox


    def _handle_error(self, errmsg, emailmsg, cs):
        logger.error(errmsg)
        logger.error(f"This error occurred while decoding with {cs} charset.")
        logger.error(f"These charsets were found in this email: {self._get_charsets(emailmsg)}")
        logger.error(f"This is the subject: {emailmsg['subject']}")
        logger.error(f"This is the sender: {emailmsg['From']}")

        return


    def _verify(self):
        """
        Make sure the environment is setup
        """
        if self.user == None or self.password == None:
            logger.error("Did not find all redmine configs: IMAP_USER, IMAP_PW (source ~/.imap).")
            return False

        return True


    def _extract_email_body(self, payload):
        """
        Extracts the visible text content from HTML payload.
        """
        try:
            soup = BeautifulSoup(payload, 'html.parser')

            body_text = soup.get_text(separator='\n').strip()
            logger.debug("Extracted body text:", body_text)
            logger.debug(type(body_text))
            return body_text
        except Exception as e:
            logger.error(f"Failed to parse HTML: {e}")
            return ""
