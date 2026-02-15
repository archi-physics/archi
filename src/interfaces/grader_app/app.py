import base64
import csv
import datetime
import glob
import json
import os
import random
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
import openai
import psycopg2
import psycopg2.extras
import yaml
from flask import (Flask, Response, flash, redirect, render_template, request,
                   session, url_for)
from flask_cors import CORS
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_session import Session  # Use Flask-Session for server-side sessions
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
# Imports for TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import \
    cosine_similarity as sklearn_cosine_similarity

from src.archi.archi import archi
from src.data_manager.data_manager import DataManager
from src.utils.env import read_secret
from src.utils.logging import get_logger
from src.utils.sql import (SQL_INSERT_CONVO,
                           SQL_INSERT_FEEDBACK, SQL_INSERT_TIMING,
                           SQL_QUERY_CONVO)
from src.utils.config_access import get_full_config, get_global_config, get_services_config

logger = get_logger(__name__)

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

class ImageToTextWrapper:
    def __init__(self):
        self.config = get_full_config()
        self.global_config = get_global_config()
        self.services_config = get_services_config()
        self.data_path = self.global_config["DATA_PATH"]
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        self.lock = Lock()

        # initialize image processing chain
        grader_cfg = self.services_config.get("grader_app", {})
        image_prompts = grader_cfg.get("prompts", {}).get("image_processing", {})
        self.image_processor = archi(
            pipeline="ImageProcessingPipeline",
            default_provider=grader_cfg.get("default_provider"),
            default_model=grader_cfg.get("default_model"),
            prompt_overrides=image_prompts,
        )

    def __call__(self, images: List[str]) -> str:
        """
        Main call method: accepts list of base64 images, returns extracted text.
        """
        self.lock.acquire()
        try:
            text = self.image_processor(images) # from __call__ of the model client
            
            # LOGGING AND DATABASE !!! (later)

        except Exception as e:
            logger.error(f"Failed to convert image to text: {str(e)}")
            text = "Error processing images"

        finally:
            self.lock.release()
            if self.cursor is not None:
                self.cursor.close()
            if self.conn is not None:
                self.conn.close()
                
        return text

        # FUNCTIONS FOR POSTGRES INSERTS
    
        # FUNCTIONS FOR LOGGING


class GradingWrapper:
    def __init__(self):
        self.config = get_full_config()
        self.global_config = get_global_config()
        self.services_config = get_services_config()
        self.data_path = self.global_config["DATA_PATH"]

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        self.lock = Lock()

        # initialize grading chain
        grader_cfg = self.services_config.get("grader_app", {})
        grading_prompts = grader_cfg.get("prompts", {}).get("grading", {})
        self.grader = archi(
            pipeline="GradingPipeline",
            default_provider=grader_cfg.get("default_provider"),
            default_model=grader_cfg.get("default_model"),
            prompt_overrides=grading_prompts,
        )


    ##################

    #     much of the below will be moved either to models.py or to the chain.py file or something.
    
    #     especially, most obviously, get_evaluation

    ##################
    def __call__(self, student_solution: str, official_explanation: str, additional_comments: str = "") -> str:
        """
        Main grading pipeline: run summary → analysis → final decision.
        Returns final evaluation text.
        """
        self.lock.acquire()
        try:
            final_decision = self.grader(
                submission_text=student_solution,
                rubric_text=official_explanation,
                additional_comments=additional_comments
            )
        except Exception as e:
            logger.error(f"Failed to grade submission: {str(e)}")
            final_decision = "Error during grading pipeline"
        finally:
            self.lock.release()
            if self.cursor is not None:
                self.cursor.close()
            if self.conn is not None:
                self.conn.close()
        return final_decision




class FlaskAppWrapper(object):
    def __init__(self, app: Flask, **configs):
        self.app = app
        self.configs(**configs)
        self.config = get_full_config()
        self.global_config = get_global_config()
        self.services_config = get_services_config()
        self.data_path = self.global_config["DATA_PATH"]

        # session config
        self.app.secret_key = 'your_secret_key' # read_secret later... debugging
        self.app.config['SESSION_TYPE'] = 'filesystem'
        self.app.config['SESSION_FILE_DIR'] = '/tmp/flask_session' # writeable directory in container for session files
        Session(self.app)

        # setup login manager with Flask-Login
        self.login_manager = LoginManager(self.app)
        self.login_manager.login_view = 'login'
        self.login_manager.user_loader(self.load_user)

        # load users
        self.csv_filename = "/root/archi/users.csv" # or wherever you decide to put this file...
        self.users_db = self.load_users(self.csv_filename)
        
        # load admin password
        self.admin_password = "222222" # read_secret("ADMIN_PASSWORD")

        # store postgres connection info
        self.pg_config = {
            "password": read_secret("PG_PASSWORD"),
            **self.services_config["postgres"],
        }
        self.conn = None
        self.cursor = None

        # track current model and pipeline for conversation logging
        self.current_model_used = self._extract_model_name()
        self.current_pipeline_used = "GradingPipeline"

        # WRAPPERS 
        self.image_processor = ImageToTextWrapper()
        self.grader = GradingWrapper()

        # enable CORS
        CORS(self.app)

        # add endpoints
        self.add_routes()
        ########### templates not static...







    def configs(self, **configs):
        for config, value in configs:
            self.app


    def load_user(self, user_id):
        if user_id in self.users_db:
            return User(email=user_id)
        return None

    def load_users(self, csv_filename):
        """Load users from a CSV file."""
        users = {}
        with open(csv_filename, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                normalized = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}
                email = normalized.get("mit email")
                unique_code = normalized.get("unique code")
                if email and unique_code:
                    users[email.lower()] = {"access_code": unique_code}
        return users

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods, *args, **kwargs)

    def add_routes(self):
        self.add_endpoint("/", "welcome", login_required(self.welcome))
        self.add_endpoint("/login", "login", self.login, methods=["GET", "POST"])
        self.add_endpoint("/logout", "logout", login_required(self.logout))
        self.add_endpoint("/problem/<int:problem_number>", "problem", login_required(self.problem), methods=["GET", "POST"])
        self.add_endpoint("/problem/<int:problem_number>/finalize", "finalize_submission", login_required(self.finalize_submission), methods=["POST"])
        self.add_endpoint("/reset_attempts/<int:problem_number>", "reset_attempts", login_required(self.reset_attempts), methods=["POST"])
        self.add_endpoint("/admin", "admin_controls", login_required(self.admin_controls), methods=["GET", "POST"])
        self.add_endpoint("/upload_rubrics", "upload_rubrics", login_required(self.upload_rubrics), methods=["POST"])
        self.add_endpoint("/thankyou", "thankyou", self.thankyou)
        self.add_endpoint("/save_comment", "save_comment", login_required(self.save_comment), methods=["POST"])
        self.add_endpoint("/download_evaluation/<submission_id>", "download_evaluation", login_required(self.download_evaluation))

    
    def welcome(self):
        is_mobile, accessibility = self.get_device_flags()
        welcome_file = os.path.join(self.data_path, "welcome_message.txt")
        try:
            with open(welcome_file, 'r') as f:
                welcome_message = f.read()
        except FileNotFoundError:
            welcome_message = "Welcome! Please proceed with your submission."
        return render_template(
            "welcome.html",
            welcome_message=welcome_message,
            email=current_user.id,
            is_mobile=is_mobile,
            accessibility=accessibility,
        )

    def login(self):
        is_mobile, accessibility = self.get_device_flags()
        if request.method == 'POST':
            email = request.form.get('email').strip().lower()
            code = request.form.get("code").strip()
            if not email.endswith('@mit.edu'):
                flash("Please use your MIT email address.")
                return redirect(url_for('login'))
            user_data = self.users_db.get(email)
            if not user_data:
                flash("Email not found. Please use your MIT email.")
                return redirect(url_for('login'))
            if user_data['access_code'] == code:
                user = User(email=email)
                login_user(user)
                return redirect(url_for('welcome'))
            else:
                flash("Invalid access code. Please try again.")
                return redirect(url_for('login'))
        return render_template(
            "login.html",
            is_mobile=is_mobile,
            accessibility=accessibility,
        )

    def logout(self):
        logout_user()
        return redirect(url_for('login'))

    def problem(self, problem_number):
        """ 
        Handle problem submission and display.
        """
    
        is_mobile, accessibility = self.get_device_flags()

        total_problems = self.get_total_problems()
        logger.info(f"Total problems: {total_problems}")


        if problem_number < 1 or problem_number > total_problems:
            return "Invalid problem number", 400

        if request.method == 'GET':
            if self.count_attempts(current_user.id, problem_number) > 0:
                update_grades_cache()
                records = get_grades_cache()
                record = next((r for r in records if r[1].strip().lower() == current_user.id.lower() and int(r[6]) == problem_number), None)
                if record:

                    try:
                        earned_total, max_total = map(int, record[2].split('/'))
                    except Exception:
                        earned_total, max_total = 0, 0
                    
                    final_score_eval = float(record[11])
                    final_score_combined = float(record[12])
                    percentage = final_score_eval

                    if percentage <= 40:
                        performance_message = "unsatisfactory performance"
                    elif percentage <= 80:
                        performance_message = "satisfactory performance"
                    else:
                        performance_message = "excellent work"

                    return render_template(
                        "result.html",
                        performance_message=performance_message,
                        detailed_evaluation=record[4],
                        student_evaluation=record[4],
                        final_score_skeleton=record[7] if len(record) > 7 else "N/A",
                        handwritten_explanation="",
                        handwritten_images=record[3].split("||"),
                        problem_number=problem_number,
                        earned_total=earned_total,
                        max_total=max_total,
                        total_problems=total_problems,
                        final_score_eval=final_score_eval,
                        final_score_combined=final_score_combined,
                        submission_id=record[0],
                        is_mobile=is_mobile, accessibility=accessibility),

            else:
                rubric_text = self.get_rubric(problem_number)
                problem_name = f"Problem {problem_number}"
                if rubric_text:
                    for line in rubric_text.splitlines():
                        stripped_line = line.strip()
                        if stripped_line and not all(ch == '-' for ch in stripped_line):
                            problem_name == stripped_line
                            break
                return render_template(
                    "index.html",
                    problem_number=problem_number,
                    problem_name=problem_name,
                    attemps_exceeded=False,
                    total_problems=total_problems,
                    is_mobile=is_mobile,
                    accessibility=accessibility,
                )

        if request.method == 'POST':
            if self.count_attempts(current_user.id, problem_number) > 0:
                flash("You have already submitted this problem.")
                return redirect(url_for('problem', problem_number=problem_number))
            
            files = request.files.getlist('handwritten_image')
            if not files or len(files) == 0:
                return "No images provided", 400

            # image to "text" here, then will pass to llm
            base64_images = []
            for file in files:
                if file.filename == '':
                    continue
                data = file.read()
                base64_images.append(base64.b64encode(data).decode('utf-8'))
            if not base64_images:
                return "No valid images provided", 400

            submission_id = str(uuid.uuid4())

            official_explanation = self.get_rubric(problem_number)
            if official_explanation is None:
                return f"Rubric file not found for problem {problem_number}", 404

            #################################

            # IMAGE PROCESSING CALLED HERE !!!
            # TODO this is only for one image (?)
            image_processor_output = self.image_processor(base64_images)
            answer_payload = image_processor_output.answer
            if isinstance(answer_payload, dict):
                raw_response = answer_payload.get("text", [""])[0]
            else:
                raw_response = str(answer_payload)
            normalized_response = self.normalize_llm_response(raw_response)

            #################################

            if '--- Notes ---' in normalized_response:
                parts = normalized_response.split('--- Notes ---', 1)
                response_final = parts[0].strip()
            else:
                response_final = normalized_response

            response_final = self.clean_latex_output(response_final)

            session['submission_id'] = submission_id
            session['base64_images'] = base64_images
            session['problem_number'] = problem_number
            session['response_final'] = response_final

            total_problems = self.get_total_problems()

            return render_template(
                "conversion_approval.html",
                conversion_text=response_final,
                base64_images=base64_images,
                problem_number=problem_number,
                total_problems=total_problems,
                is_mobile=is_mobile,
                accessibility=accessibility,
            )

    def clean_latex_output(self, text: str) -> str:
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*', '', text)
        
        text = re.sub(r'\\begin{.*?}.*?\\end{.*?}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section{.*?}', '', text)
        text = re.sub(r'\\subsection{.*?}', '', text)
        
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # fix double backslashes in math delimiters for MathJax
        text = text.replace('\\\\[', '\\[').replace('\\\\]', '\\]')
        text = text.replace('\\\\(', '\\(').replace('\\\\)', '\\)')
        
        return text.strip()
    
    def normalize_llm_response(self, response: str) -> str:
        response = response.strip()

        if response.startswith("[") and response.endswith("]"):
            # Handle both single or double quoted list strings: ['...'] or ["..."]
            if (response.startswith("['") and response.endswith("']")) or \
            (response.startswith('["') and response.endswith('"]')):
                response = response[2:-2]  # Remove [' or [" and '] or "]
                response = response.replace("\\n", "\n").replace("\\t", "\t")
                return response

        return response


    def finalize_submission(self, problem_number):
        """
        Finalize and evaluate a student submission.
        """

        submission_id = session.get('submission_id')
        base64_images = session.get('base64_images')
        original_conversion_text = session.get('response_final')
        approved_text = request.form.get('approved_text', original_conversion_text)
        additional_comments = request.form.get('additional_comments', "")

        final_student_solution = approved_text
        if additional_comments.strip():
            final_student_solution += f"\n\nAdditional Comments:\n" + additional_comments

        official_explanation = self.get_rubric(problem_number)
        if official_explanation is None:
            return f"Rubric file not found for problem {problem_number}", 500

        #################################

        # GRADING CALLED HERE !!!

        grading_result = self.grader(final_student_solution, official_explanation, additional_comments)
        print(grading_result)
        grading_result_score = grading_result["final_grade"]

        logger.info(f"Grading result score: {grading_result_score}")

        earned_total, max_total = self.calculate_total_score(grading_result_score)
        final_score_eval = (earned_total / max_total) * 100 if max_total > 0 else 0

        percentage = final_score_eval
        if percentage <= 40:
            performance_message = "unsatisfactory performance"
        elif percentage <= 80:
            performance_message = "satisfactory performance"
        else:
            performance_message = "excellent work"

        # need to return:
        # - performance_message
        # - detailed_evaluation
        # - student_evaluation
        # - final_score_skeleton
        # - earned_total
        # - final_score_eval
        # - avg_earned
        # - score_std
        

        # and also write to grades.csv (or later postgres) -- and do differently, so not referencing record[int] constantly


        #################################

        return render_template(
            "result.html",
            performance_message=performance_message,
            detailed_evaluation=grading_result_score,
            student_evaluation="todo: student_evaluation",
            final_score_skeleton=222, # not running non-AI evaluation yet
            handwritten_explanation=final_student_solution+"\n\nAdditional comments:\n\n"+additional_comments,
            handwritten_images=base64_images,
            problem_number=problem_number,
            earned_total=earned_total,
            max_total=max_total,
            total_problems=self.get_total_problems(),
            final_score_eval=final_score_eval,
            avg_earned=222, # not running multiple evaluations yet
            score_std=222, # not running multiple evaluations yet
            submission_id=submission_id,
            is_mobile=self.get_device_flags()[0],
            accessibility=self.get_device_flags()[1]
        )




    def reset_attempts(self, problem_number):
        admin_password = request.form.get('admin_password', '')
        ####### MAKE SURE TO DO THIS PASSWORD BULLSHIT CLEANLY !!!
        if admin_password != self.admin_password:
            flash("Unauthorized: Incorrect admin password.")
            return redirect(url_for('problem', problem_number=problem_number))
        
        student_email = request.form.get('student_email', '').strip().lower()
        if not student_email:
            flash("Please provide a student email.")
            return redirect(url_for('admin_controls'))

        try:
            if os.path.exists(self.csv_filename):
                with open(self.csv_filename, 'r', newline='') as f:
                    reader = csv.reader(f)
                    records = list(reader)
                
                filtered_records = [
                    record for record in records
                    if not (record[1].strip().lower() == student_email and int(record[6]) == problem_number)
                ]

                with open(self.csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(filtered_records)
            flash(f"Successfully reset attempts for {student_email} on Problem {problem_number}.")

        except Exception as e:
            logger.error(f"Error resetting attempts: {str(e)}")
            flash("Error resetting attempts.")

        update_grades_cache()
        return redirect(url_for('admin_controls'))

    

    def admin_controls(self):

        is_mobile, accessibility = self.get_device_flags()

        if request.method == 'POST':
            admin_password = request.form.get('admin_password', '')
            if admin_password != self.admin_password:
                flash("Unauthorized: Incorrect admin password.")
                return redirect(url_for('admin_controls'))

            return render_template(
                "admin_controls.html",
                is_mobile=is_mobile,
                accessibility=accessibility,
            )

        return render_template(
            "admin_login.html",
            is_mobile=is_mobile,
            accessibility=accessibility,
        )



    def upload_rubrics(self):
        
        is_mobile, accessibility = self.get_device_flags()

        admin_password = request.form.get('admin_password', '')
        if admin_password != self.admin_password:
            flash("Unauthorized: Incorrect admin password.")
            return redirect(url_for('admin_controls'))
        
        files = request.files.getlist('rubric_files')
        if not files:
            flash("No files uploaded.")
            return redirect(url_for('admin_controls'))

        for file in files:
            filename = file.filename
            if not filename.endswith('.txt'):
                continue
            
            ######## CORRECTLY USING DATA_PATH AS YOU SHOULD IN METHODS BELOW
            save_path = os.path.join(self.data_path, filename)
            file.save(save_path)
            problem_number = re.findall(r'\d+', filename)

            if problem_number:
                self.get_rubric.cache.clear()

        flash("Rubrics uploaded successfully.")

        return redirect(url_for('admin_controls'))




    def thankyou(self):

        is_mobile, accessibility = self.get_device_flags()

        ######## CORRECTLY USING DATA_PATH AS YOU SHOULD IN METHODS BELOW
        message_file = os.path.join(self.data_path, "thankyou_message.txt")

        try:
            with open(message_file, 'r') as f:
                thank_you_message = f.read()
        except FileNotFoundError:
            thank_you_message = "Thank you for your submission!"
        
        return render_template(
            "thankyou.html",
            thank_you_message=thank_you_message,
            is_mobile=is_mobile,
            accessibility=accessibility,
        )


    def save_comment(self):

        is_mobile, accessibility = self.get_device_flags()

        submission_id = request.form.get('submission_id')
        comment_text = request.form.get('comment')

        if not submission_id or not comment_text:
            flash("Submission ID or comment is missing.")
            return redirect(request.referrer)

        # WHAT IS THIS FILE, WHERE DOES IT COME FROM, AND DOES IT GET PASSED INITIAILLY OR NOT ***(DATA_PATH OR NOT)***
        comment_file = "comments.csv"

        timestamp = datetime.datetime.now().isoformat()

        with open(comment_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([submission_id, current_user.id, comment_text, timestamp])
        flash("Comment saved successfully.")
        return redirect(request.referrer)



    def download_evaluation(self, submission_id):
        
        evaluation_record = None

        if os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == submission_id and row[1].strip().lower() == current_user.id.lower():
                        evaluation_record = row
                        break

        if not evaluation_record:
            flash("Evaluation record not found.")
            return redirect(url_for('welcome'))

        details = f"Submission ID: {evaluation_record[0]}\n"
        details += f"Student Email: {evaluation_record[1]}\n"
        details += f"Score: {evaluation_record[2]}\n"
        details += f"Timestamp: {evaluation_record[5]}\n"
        details += f"Problem Number: {evaluation_record[6]}\n\n"
        details += "Detailed Evaluation:\n"
        details += f"{evaluation_record[4]}\n\n"
        details += "Final Scores:\n"
        details += f"AI Evaluation Score: {evaluation_record[11]}\n"
        comment = ""
        if os.path.exists("comments.csv"):
            with open("comments.csv", "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == submission_id and row[1].strip().lower() == current_user.id.lower():
                        comment = row[2]
                        break
        if comment:
            details += f"\nComment: {comment}\n"

        return Response(details, mimetype="text/plain",
                        headers={"Content-Disposition": f"attachment;filename=evaluation_{submission_id}.txt"})



    def run(self, **kwargs):
        self.app.run(**kwargs)


    def _extract_model_name(self):
        """Extract model name from config for conversation logging."""
        try:
            # Navigate through config to find the model being used
            pipelines_config = self.config.get("pipelines", {})
            grading_pipeline = pipelines_config.get("GradingPipeline", {})
            models = grading_pipeline.get("models", [])
            if models and len(models) > 0:
                first_model = models[0]
                if isinstance(first_model, dict):
                    return first_model.get("name", "unknown")
                return str(first_model)
        except Exception:
            pass
        return "unknown"


    def get_device_flags(self):
        is_mobile = "iphone" in request.user_agent.string.lower()
        accessibility = request.args.get("accessibility", "0") == "1"
        return is_mobile, accessibility
    
    def calculate_total_score(self, evaluation_text):
        cleaned_text = evaluation_text.replace("**", "")
        scores = re.findall(
            r"Score:\s*(?:\\textbf\{)?\s*([\d]+)\s*(?:\})?\s*/\s*(?:\\textbf\{)?\s*([\d]+)\s*(?:\})?",
            cleaned_text,
            flags=re.UNICODE
        )
        earned_total = sum(int(earned) for earned, _ in scores)
        max_total = sum(int(max_points) for _, max_points in scores)
        return earned_total, max_total


    def get_total_problems(self):
        return self.config["services"]['grader_app'].get('num_problems')

    def count_attempts(self, user_email, problem_number):
        count = 0
        records = get_grades_cache()
        for row in records:
            if row and row[1].strip().lower() == user_email.lower() and int(row[6]) == problem_number:
                count += 1
        return count


    @lru_cache(maxsize=None)
    @staticmethod
    def get_rubric(self, problem_number):
        rubric_file = f"/root/archi/solution_with_rubric_{problem_number}.txt"
        try:
            with open(rubric_file, "r") as f:
                return f.read()
        except FileNotFoundError:
            return None

    # in case want to switch to API endpoints later...
    '''def image_processor_endpoint(self):
        images = request.json.get('images')  # List of base64
        result = self.image_processor(images)
        return jsonify({'result': result})

    def grader_endpoint(self):
        official_explanation = request.json.get('official_explanation')
        student_solution = request.json.get('student_solution')
        additional_comments = request.json.get('additional_comments', "")
        result = self.grader(official_explanation, student_solution, additional_comments)
        return jsonify({'result': result})'''


    ##########

    # define methods for rendering html templates here

    ##########



class User(UserMixin):
        def __init__(self, email):
            self.id = email


#############################################################################

####### GLOBAL GRADES CACHE #######

# this is a simple cache for grades, stored in a csv file
# -->> upgrade to postgres! but maybe not urgent and can just do later.......

grades_cache = None
def load_grades_cache():
    global grades_cache
    if os.path.exists("grades.csv"):
        with open("grades.csv", "r", newline="") as f:
            reader = csv.reader(f)
            grades_cache = list(reader)
    else:
        grades_cache = []
def get_grades_cache():
    global grades_cache
    if grades_cache is None:
        load_grades_cache()
    return grades_cache
def update_grades_cache():
    load_grades_cache()
