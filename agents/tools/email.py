import os
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid

from dotenv import load_dotenv
from datetime import datetime

load_dotenv

drafts = {}

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

def generate_draft(recipient, subject, body):
    draft_id = str(uuid.uuid4())
    draft = {
        "id": draft_id,
        "recipient": recipient,
        "subject": subject,
        "body": body,
        "created_at": datetime.now(),
        "confirmed": False
    }
    drafts[draft_id] = draft
    return draft


def send_email(draft_id):
    draft = drafts.get(draft_id)
    if not draft or not draft.get("confirmed"):
        raise ValueError("Draft not found or not confirmed")
    
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = draft["recipient"]
    msg["Subject"] = draft["subject"]
    
    msg.attach(MIMEText(draft["body"], "plain"))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(msg["From"], msg["To"], msg.as_string())
        server.quit()
        del drafts[draft_id]
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        del drafts[draft_id]
