import os
import re
import imaplib
import email
import requests
import smtplib
import streamlit as st
from email.header import decode_header
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("GMAIL_EMAIL")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
PERSON_NAME = os.getenv("PERSON_NAME", "Suyash")
BRAIN_API_URL = os.getenv("BRAIN_API_URL", "http://127.0.0.1:8000/clone/process")


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
            disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="ignore")

    return body.strip()


def extract_email_address(sender):
    match = re.search(r"<(.+?)>", sender)
    return match.group(1) if match else sender


def fetch_unread_emails():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, APP_PASSWORD.replace(" ", ""))
    mail.select("inbox")

    status, response = mail.search(None, "UNSEEN")
    email_ids = response[0].split()

    emails = []

    for e_id in email_ids:
        status, data = mail.fetch(e_id, "(RFC822)")
        raw_email = data[0][1]

        msg = email.message_from_bytes(raw_email)

        subject = decode_mime_words(msg.get("Subject"))
        sender = decode_mime_words(msg.get("From"))
        body = extract_body(msg)

        emails.append({
            "sender": sender,
            "subject": subject,
            "body": body
        })

    mail.logout()
    return emails


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
        "clone_draft": "",
        "confidence_score": 0,
        "reasoning": response.text,
        "status": "error"
    }


def send_email_reply(to_email, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = EMAIL
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL, APP_PASSWORD.replace(" ", ""))
        server.send_message(msg)


st.set_page_config(
    page_title="Digital Clone Gmail Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Digital Clone Gmail Dashboard")
st.caption("Unread emails → Persona memory → AI draft → Auto-send / Manual approval")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Clone", PERSON_NAME)

with col2:
    st.metric("Email", EMAIL)

with col3:
    st.metric("Auto-send rule", "Confidence ≥ 90")

st.divider()

if st.button("🔍 Check New Emails"):
    with st.spinner("Checking Gmail inbox..."):
        emails = fetch_unread_emails()

    if not emails:
        st.success("No new unread emails found.")
    else:
        st.success(f"Found {len(emails)} unread email(s).")

        for index, mail_item in enumerate(emails):
            sender = mail_item["sender"]
            subject = mail_item["subject"]
            body = mail_item["body"]

            with st.container(border=True):
                st.subheader(f"📩 {subject}")
                st.write("**From:**", sender)
                st.write("**Message:**")
                st.info(body[:1000])

                with st.spinner("Generating clone response..."):
                    brain_response = send_to_brain(sender, subject, body)

                draft = brain_response.get("clone_draft", "")
                confidence = brain_response.get("confidence_score", 0)
                reasoning = brain_response.get("reasoning", "")
                status = brain_response.get("status", "pending")

                st.write("### 🤖 Clone Response")
                st.write(draft)

                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Confidence", confidence)

                with c2:
                    if status == "auto_sent":
                        st.success("Status: Auto Sent")
                    elif status == "pending":
                        st.warning("Status: Pending Approval")
                    else:
                        st.error("Status: Error")

                st.write("**Reasoning:**")
                st.caption(reasoning)

                to_email = extract_email_address(sender)

                if status == "auto_sent" and draft:
                    try:
                        send_email_reply(to_email, subject, draft)
                        st.success(f"✅ Auto reply sent to {to_email}")
                    except Exception as e:
                        st.error(f"Auto-send failed: {e}")

                elif draft:
                    if st.button(f"✅ Manually Send Reply {index + 1}"):
                        try:
                            send_email_reply(to_email, subject, draft)
                            st.success(f"Reply sent to {to_email}")
                        except Exception as e:
                            st.error(f"Sending failed: {e}")
