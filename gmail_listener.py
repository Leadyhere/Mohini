import os
import imaplib
import email
import time
import requests
from email.header import decode_header
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

def _clean_env_value(name):
    value = os.getenv(name)

    if value is None:
        return None

    value = value.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()

    return value


EMAIL = _clean_env_value("GMAIL_EMAIL")
APP_PASSWORD = _clean_env_value("GMAIL_APP_PASSWORD")
BRAIN_API_URL = os.getenv("BRAIN_API_URL", "http://127.0.0.1:8000/clone/process")
PERSON_NAME = os.getenv("PERSON_NAME", "Suyash")


def validate_config():
    missing_vars = []

    if not EMAIL:
        missing_vars.append("GMAIL_EMAIL")

    if not APP_PASSWORD:
        missing_vars.append("GMAIL_APP_PASSWORD")

    if missing_vars:
        missing_list = ", ".join(missing_vars)
        raise RuntimeError(
            f"Missing required environment variables in {ENV_PATH}: {missing_list}"
        )

    if "@" not in EMAIL:
        raise RuntimeError("GMAIL_EMAIL must be a valid Gmail address.")

    if len(APP_PASSWORD.replace(" ", "")) != 16:
        raise RuntimeError(
            "GMAIL_APP_PASSWORD must be a 16-character Gmail App Password. "
            "Do not use your normal Gmail password."
        )


def decode_mime_words(text):
    if text is None:
        return ""

    decoded_parts = decode_header(text)
    result = ""

    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            result += part.decode(encoding or "utf-8", errors="ignore")
        else:
            result += str(part)

    return result


def extract_body(msg):
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)

                if payload:
                    body = payload.decode(errors="ignore")
                    break
    else:
        payload = msg.get_payload(decode=True)

        if payload:
            body = payload.decode("utf-8", errors="ignore")

    return body.strip()


def send_to_brain(sender, subject, body):
    payload = {
        "person_name": PERSON_NAME,
        "sender_name": sender,
        "message_text": f"Subject: {subject}\n\nBody: {body}"
    }

    response = requests.post(BRAIN_API_URL, json=payload)

    if response.status_code == 200:
        return response.json()

    return {
        "error": response.text
    }


def listen_for_emails():
    try:
        validate_config()
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, APP_PASSWORD.replace(" ", ""))
        mail.select("inbox")

        status, response = mail.search(None, "UNSEEN")
        email_ids = response[0].split()

        if not email_ids:
            print("Listening... No new unread emails.")
        else:
            for e_id in email_ids:
                status, data = mail.fetch(e_id, "(RFC822)")
                raw_email = data[0][1]

                msg = email.message_from_bytes(raw_email)

                subject = decode_mime_words(msg.get("Subject"))
                sender = decode_mime_words(msg.get("From"))
                body = extract_body(msg)

                print("\nNEW EMAIL DETECTED")
                print("From:", sender)
                print("Subject:", subject)
                print("Body:", body[:300])

                brain_response = send_to_brain(sender, subject, body)

                print("\nCLONE RESPONSE")
                print("Draft:", brain_response.get("clone_draft"))
                print("Confidence:", brain_response.get("confidence_score"))
                print("Reasoning:", brain_response.get("reasoning"))
                print("Status:", brain_response.get("status"))

        mail.logout()

    except imaplib.IMAP4.error as e:
        error_text = str(e)

        if "Application-specific password required" in error_text:
            print(
                "Error: Gmail rejected the login. "
                "Use a 16-character Google App Password for GMAIL_APP_PASSWORD, "
                "not your normal Gmail password."
            )
            print(
                "Also make sure 2-Step Verification is enabled on that Google account "
                "and paste the App Password into .env without quotes."
            )
        else:
            print("Error:", e)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    while True:
        listen_for_emails()
        time.sleep(60)
