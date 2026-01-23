import datetime
import os
import re
import traceback

import psycopg2
import psycopg2.extras
from redminelib import Redmine as RedmineClient

from src.a2rchi.a2rchi import A2rchi
from src.data_manager.collectors.utils.index_utils import CatalogService
from src.data_manager.data_manager import DataManager
from src.interfaces.redmine_mailer_integration.utils import sender
from src.utils.runtime_config_loader import load_runtime_config
from src.utils.env import read_secret
from src.utils.logging import get_logger
from src.utils.sql import SQL_INSERT_CONVO

logger = get_logger(__name__)

# DEFINITIONS
A2RCHI_PATTERN = '-- A2rchi --'


class RedmineAIWrapper:
    """
    Wrapper which holds functionality for the redminebot. Way of interaction
    between redmine and A2rchi core.
    """

    def __init__(self):

        # initialize data manager
        self.data_manager = DataManager(run_ingestion=False)

        # configs
        self.config = load_runtime_config()
        self.global_config = self.config["global"]
        self.services_config = self.config["services"]
        self.redmine_config = self.services_config.get("redmine_mailbox", {})
        self.data_path = self.global_config["DATA_PATH"]

        # agent
        self.a2rchi = A2rchi(pipeline=self.redmine_config.get("pipeline", "CMSCompOpsAgent"))

        # postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        self.config_id = 1 # TODO: make dynamic a la chat_app/app.py

    def prepare_context_for_storage(self, source_documents):
        
        # load the present list of sources
        sources = CatalogService.load_sources_catalog(self.data_path, self.pg_config)

        num_retrieved_docs = len(source_documents)
        context = ""
        if num_retrieved_docs > 0:
            for k in range(num_retrieved_docs):
                document = source_documents[k]
                link = document.metadata.get('url', 'No URL')
                multiple_newlines = r'\n{2,}'
                content = re.sub(multiple_newlines, '\n', document.page_content)
                context += f"Source {k+1}: {document.metadata.get('title', 'No Title')} ({link})\n\n{content}\n\n\n\n"

        return link, context

    def insert_conversation(self, issue_id, user_message, a2rchi_message, link, a2rchi_context, ts):
        logger.info("Storing interaction to postgres")

        service = "Redmine"

        insert_tups = (
            [
                # (service, issue_id, sender, content, context, ts) -- same ts for both just to have, not as interested in timing info for redmine service...
                (service, issue_id, "User", user_message, '', '', ts, self.config_id),
                (service, issue_id, "A2rchi", a2rchi_message, link, a2rchi_context, ts, self.config_id),
            ]
        )

        # create connection to database
        self.conn = psycopg2.connect(**self.pg_config)
        self.cursor = self.conn.cursor()
        psycopg2.extras.execute_values(self.cursor, SQL_INSERT_CONVO, insert_tups)
        self.conn.commit()

        # clean up database connection state
        self.cursor.close()
        self.conn.close()
        self.cursor, self.conn = None, None


    def __call__(self, history, issue_id):
        # create formatted history
        reformatted_history = []
        for entry in history:
            if "ISSUE_ID:" in entry[1]:
                role = "Expert"
            else:
                role = "A2rchi"
            message = RedmineAIWrapper.get_substring_between(entry[1],"\n\nRe:","\r\nOn ")
            reformatted_history.append((role,message))
        reformatted_history[0] = ("Expert", reformatted_history[0][1])
        reformatted_history[-1] = ("User", reformatted_history[-1][1])

        # execute chain and get answer
        result = self.a2rchi(history=reformatted_history)
        answer = result.answer

        # prepare other information for storage
        history = "Question: " + reformatted_history[-1][1] + "\n\n\n\nHistory:\n\n" + "\n\n".join(post[0] + ": " + post[1] for post in reversed(reformatted_history[:-1]))
        link, context = self.prepare_context_for_storage(result.source_documents)
        ts = datetime.datetime.now()

        self.insert_conversation(issue_id, history, answer, link, context, ts)
        
        return answer

    @staticmethod
    def get_substring_between(text, start_word, end_word):
        """
        Small helper function. Return everything (not including) between the 
        start_word and the end_word if start_word and end_word exist. Otherwise
        it does nothing
        """
        start_index = text.find(start_word)
        end_index = text.find(end_word)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return text[start_index + len(start_word):end_index].strip()
        else:
            return text


class Redmine:
    'A class to describe the redmine system.'

    def __init__(self, name):
        """
        Give it a name and generate a conncetion to the database (should be a singleton).
        """
        self.name = name             # to identify
        self.redmine = None
        self.smtp = sender.Sender()
        self.user = None
        self.project = None
        self.ai_wrapper = None
        if self.name != "Redmine_Helpdesk_Mail":
            logger.info("Loading AI wrapper for Redmine service")
            self.ai_wrapper = RedmineAIWrapper()

        # read configuration for Redmine mailbox service
        config = load_runtime_config()
        services_config = config.get("services", {}) if isinstance(config, dict) else {}
        redmine_mailbox_config = services_config.get("redmine_mailbox", {})

        self.redmine_url = redmine_mailbox_config.get("url")
        self.redmine_project = redmine_mailbox_config.get("project")

        if not self.redmine_url or not self.redmine_project:
            logger.warning(
                "Redmine mailer configuration missing services.redmine_mailbox.url/project; skipping initialisation."
            )
            return

        try:
            self.redmine_user = read_secret("REDMINE_USER")
            self.redmine_pw = read_secret("REDMINE_PW")
        except FileNotFoundError as error:
            logger.warning(
                "Redmine credentials not found in secrets; skipping Redmine mailer initialisation.",
                exc_info=error,
            )
            return

        # make sure to open redmine access
        if self._verify():
            self.redmine = self._connect()
            self.user = self.redmine.user.get('current')
            self.load()

        # Load all the status, tracker, and priority ids
        statuses = self.redmine.issue_status.all()
        self.status_dict = dict()     # keys = status name, values = status_id
        for s in statuses:
            self.status_dict[s.name] = s.id

        trackers = self.redmine.tracker.all()
        self.tracker_dict = dict()
        for t in trackers:
            self.tracker_dict[t.name] = t.id

        priorities = self.redmine.enumeration.filter(resource="issue_priorities")
        self.priorities_dict = dict()
        for p in priorities:
            self.priorities_dict[p.name] = p.id


            
    def add_note_to_issue(self,issue_id,note):
        """
        Adding a note to an existing issue (and move to 'feedback' status)
        """
        self.redmine.issue.update(issue_id,status_id = self.status_dict['Feedback'],notes = note)
        return

    def reopen_issue(self, issue_id, note,attachments):
        """
        Move an issues status to `In Progress` and add a note
        """
        self.redmine.issue.update(issue_id,status_id = self.status_dict['In Progress'],
                                  notes = note,uploads = attachments)
        return
    
    def get_issue_history(self,issue_id):
        """
        Extract a tuple of author and notes for this ticket
        """
        issue = self.redmine.issue.get(issue_id)
        history = [("User:", "<b>" + issue.subject + "</b> \n" + issue.description )]
        for record in issue.journals:
            user = self.redmine.user.get(record.user.id)
            note = record.notes
            if note != '' and A2RCHI_PATTERN not in note:
                history.append((user.login,note))
        return history
    
    def load(self):
        """
        Load the project that is responsible to deal with email tickets.
        """
        self.project = self.redmine.project.get(self.redmine_project)
        return

    def new_issue(self,sender,cc,subject,description,attachments):
        """
        Create a brand new issue in the redmine system
        """
        if not subject.strip():
            subject = 'EMPTY subject'
        issue = self.redmine.issue.new()
        issue.project_id = self.project.id
        issue.subject = subject
        issue.description = description
        issue.tracker_id = self.tracker_dict["Support"]
        issue.status_id = self.status_dict['New']
        issue.priority_id = self.priorities_dict['Normal']
        issue.assigned_to_id = self.user.id
        issue.watcher_user_ids = []
        #issue.parent_issue_id =
        issue.start_date = datetime.date.today()
        issue.due_date = datetime.date.today()+datetime.timedelta(1)
        issue.estimated_hours = 1
        issue.done_ratio = 0
        issue.custom_fields = [{'id': 1, 'value': sender}, {'id': 2, 'value': cc}]
        #print(issue.custom_fields)
        #issue.custom_fields = []
        #issue.uploads = [{'path': '/abs/path/to/f1'}, {'path': '/abs/path/to/f2'}]
        #print(attachments)
        issue.uploads = attachments
        issue.save()
        return issue.id

    def process_new_issues(self):
        """
        Process all issues that are assigned to me and that are in 'New' or `In Progress` status.
        """
        issue_ids = []
        for issue in self.redmine.issue.filter(assigned_to_id=self.user.id,):
            if issue.status.id == self.status_dict['New'] or issue.status.id == self.status_dict['In Progress']:
                issue_ids.append(issue.id)
                subject = f"Re:{issue.subject}"
                history = self.remove_format(f"description: {issue.description}",'pre')
                for record in issue.journals:
                    if record.notes != "":
                        history += f"\n next entry: {record.notes}"                    
                logger.info(f"History input: {history}")
                try:
                    answer = self.ai_wrapper(self.get_issue_history(issue.id), issue.id)
                except Exception as e:
                    logger.error(str(e))
                    traceback.print_exc()
                    answer = "I am sorry, I am not able to process this request at the moment. Please continue with this ticket manually."
                self.add_note_to_issue(issue.id,answer)
                logger.info(f"A2rchi's response:\n {answer}")
                self.feedback_issue(issue.id)
        logger.info("redmine.process_new_issues: %d"%(len(issue_ids)))
        return issue_ids

    def process_resolved_issues(self):
        """
        Process all issues that are in resolved mode.
        """
        issue_ids = []
        for issue in self.project.issues:
            if issue.status.id == self.status_dict['Resolved']:
                logger.info(f"Process_resolved_issues: {issue.id}")
                issue_ids.append(issue.id)
                subject = f"Re:{issue.subject}"
                to = issue.custom_fields[0]['value']
                cc = issue.custom_fields[1]['value']
                note = ''
                for record in issue.journals:
                    if record.notes and record.notes != "" and A2RCHI_PATTERN not in record.notes:
                        note = record.notes
                logger.info(f"\n TO:{to}\n CC:{cc}\n SUBJECT:{subject}\nISSUE_ID:{issue.id} (leave for reference)\n\n{note}\n\n> {issue.description}")
                note = f"\nISSUE_ID:{issue.id} (leave for reference)\n\n{note}"
                addon = issue.description.replace("\n","\n > ")
                self.smtp.send_message(to,cc,subject,f"{note}\n\nInitial request:\n > {addon}")
                self.close_issue(issue.id,note)
        logger.info("redmine.process_resolved_issues: %d"%(len(issue_ids)))
        return issue_ids
        
    def remove_format(self,string,tag):
        pattern = r"<%s>.*?</%s>"%(tag,tag)
        return re.sub(pattern,"",string,flags=re.DOTALL)
    
    def close_issue(self,issue_id,answer):
        """
        Moving the issue in the 'closed' status
        """
        self.redmine.issue.update(issue_id,status_id=self.status_dict['Closed'],
                                  notes=f'{A2RCHI_PATTERN} Resolving email was sent:\n{answer}')
        return
    
    def feedback_issue(self,issue_id):
        """
        Moving the issue in the 'feedback' status
        """
        self.redmine.issue.update(issue_id,status_id=self.status_dict['Feedback'],
                                  notes=f'{A2RCHI_PATTERN} Moved into feedback.')
        return

    def show_issue(self,issue_id):
        """
        Show issue with given id as presently in the redmine system
        """
        issue = self.project.issues.get(issue_id)
        logger.info(f"ID: {issue.id}")
        logger.info(f"Subject: {issue.subject}")
        logger.info(f"Description: {issue.description}")
        logger.info(f"Tracker: {issue.tracker} ({issue.tracker.id})")
        logger.info(f"Status: {issue.status} ({issue.status.id})")
        for record in issue.journals:
            logger.info(dir(record))
            user = self.redmine.user.get(record.user.id)
            logger.info(f" {record} ({user.login}):\n{record.notes}")
        return
            
    def show_issues(self):
        """
        Show all issues in the project
        """
        first = True
        for issue in self.project.issues:
            if first:
                first = False
                logger.info("ID status -- subject")
                logger.info("========================")
            #print(" %04d %s -- %s"%(issue.id,issue.status,issue.subject))
            self.show_issue(issue.id)
        return

    def _connect(self):
        """
        Open the redmine web site called redmine
        """
        logger.info(f"Open redmine (URL:{self.redmine_url} U:{self.redmine_user} P:*********)")
        rd = RedmineClient(self.redmine_url, username=self.redmine_user, password=self.redmine_pw)
        return rd
        
    def _verify(self):
        """
        Make sure the environment is setup
        """
        if self.redmine_url == None or self.redmine_user == None or self.redmine_pw == None:
            logger.info("Did not find all redmine configs: REDMINE_URL, REDMINE_USER, REDMINE_PW (source ~/.redmine).")
            return False
        return True
