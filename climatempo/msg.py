import bot
import smtplib
import datetime
import schedule
import time

usr = "bot.planta@gmail.com"
sub = "Subject: [Bot planta] Chuva\n"
senha = "chovechuvachovesemparar"

def envia():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(usr, senha)

    prec = bot.forecast().prec()
    msg = str(datetime.date.today())
    msg += "\n{} milimetros até o fim do dia.".format(str(round(prec, 2)))
    server.sendmail(usr, "froma.galuppo@gmail.com", sub+msg)

    server.quit()

schedule.every().day.at("06:00").do(envia)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute
