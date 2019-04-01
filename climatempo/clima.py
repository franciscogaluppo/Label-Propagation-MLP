import requests
import datetime as dt
from dateutil.parser import parse

# TODO: Documentação das funções
# TODO: Temperaturas 7 dias
# TODO: Histogramas
# TODO: Análise de correlação
# TODO: Registro de dados automático

token = "c0e087ceb8ee37852e9d31a1882bc23e"
link = "http://apiadvisor.climatempo.com.br"

def get_forecast(id=6879):
    cmd = "/api/v1/forecast/locale/"+ str(id) +"/hours/72?token=" + token
    r = requests.get(link + cmd).json()
    return(r['data'])

def print_forecast(id=6879):
    prev = get_forecast(id)
    for day in prev:
        for key in day:
            if type(day[key]) == dict:
                for param in day[key]:
                    print("{:<30}\t{:<}".format(param, day[key][param]))
            else:
                print("{:<30}\t{:<}".format(key, day[key]))
        print()

def tempo(id=6879):
    cmd = "/api/v1/weather/locale/" + str(id) + "/current?token=" + token
    r = requests.get(link + cmd).json()
    return(r['data'])

def print_tempo(id=6879):
    agora = tempo(id)
    for key in agora:
        print("{:<30}\t{:<}".format(key, agora[key]))

def get_id(cidade, estado):
    cidade.replace(" ", "+")
    cmd = "/api/v1/locale/city?name=" + cidade + "&state=" + estado +"&token=" + token
    r = requests.get(link + cmd).json()
    return(r[0]['id'])

def prec(id=6879, dia=dt.datetime.now(), dur=dt.timedelta(1)):
    r = get_forecast(id)
    soma = 0

    for horario in r:
        hora = parse(horario["date"])
        if hora >= dia and hora <= dia + dur:
            soma += horario['rain']['precipitation']

    return(soma)
