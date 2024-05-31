import imaplib
import email
from email.header import decode_header
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Gmail IMAP服务器地址
imap_server = "imap.gmail.com"
email_user = "chenalen531@gmail.com"
app_password = "gjlq xqox tyzs pytd"

# 计算一个月前的日期
one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%d-%b-%Y")
print(one_month_ago)

# 连接到Gmail IMAP服务器
mail = imaplib.IMAP4_SSL(imap_server)
try:
    mail.login(email_user, app_password)
    print("登录成功")
except imaplib.IMAP4.error as e:
    print("登录失败:", e)
    exit()

# 选择邮箱
mail.select("inbox")

# 搜索一个月内的所有邮件

status, messages = mail.search(None, f'SINCE {one_month_ago}')
if status != "OK":
    print("邮件搜索失败")
    exit()

# 获取邮件ID列表
mail_ids = messages[0].split()

# 存储邮件数据的列表
emails = []
for mail_id in mail_ids:
    status, msg_data = mail.fetch(mail_id, "(RFC822)")
    if status != "OK":
        print(f"邮件ID {mail_id} 获取失败")
        continue
    msg = email.message_from_bytes(msg_data[0][1])
    
    # 解析邮件头部信息
    sender = msg["From"]
    receiver = msg["To"]
    date = msg["Date"]
    
    # 解码邮件标题
subject = ""
try:
    subject_header = decode_header(msg["Subject"])[0]
    if isinstance(subject_header[0], bytes):
        subject = subject_header[0].decode(subject_header[1] if subject_header[1] else "utf-8")
    else:
        subject = subject_header[0]
except Exception as e:
    print("解析邮件标题时出错:", e)
# 读取邮件正文
for part in msg.walk():
    content_type = part.get_content_type()
    content_disposition = str(part.get("Content-Disposition"))
    if "attachment" not in content_disposition:
        # 解码邮件正文
        body = ""
        try:
            if content_type == "text/plain":
                # 如果是纯文本，直接使用
                body = part.get_payload(decode=True).decode()
            elif content_type == "text/html":
                # 如果是HTML，使用Beautiful Soup去除标记
                html_content = part.get_payload(decode=True).decode()
                soup = BeautifulSoup(html_content, "html.parser")
                body = soup.get_text()
        except Exception as e:
            print("解析邮件正文时出错:", e)
     # 打印邮件信息
    print("发件人:", sender)
    print("收件人:", receiver)
    print("日期:", date)
    print("主题:", subject)
    print("正文:", body)
    print("\n")
    # 存储邮件信息
    emails.append({"sender": sender, "receiver": receiver, "date": date, "subject": subject, "body": body})

    # 登出
mail.logout()

# 将邮件保存到CSV文件
df = pd.DataFrame(emails)
df.to_csv("gmail_emails.csv", index=False, encoding='utf-8-sig')

print("邮件已导出到 gmail_emails.csv")