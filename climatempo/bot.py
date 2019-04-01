import requests
import datetime as dt
from dateutil.parser import parse

# MASTER TODO: Consertar forecast.prec
# TODO: Documentação das funções
# TODO: Temperaturas 7 dias
# TODO: Histogramas
# TODO: Análise de correlação
# TODO: Registro de dados automático

token = "c0e087ceb8ee37852e9d31a1882bc23e"
link = "http://apiadvisor.climatempo.com.br"



# Previsão do tempo
def get_forecast(id=6879):
    """Faz a requisição do forecast."""

    cmd = "/api/v1/forecast/locale/"\
            +str(id) +"/hours/72?token=" + token
    
    r = requests.get(link + cmd).json()
    return(r['data'])

class forecast:
    """Classe das previsões do tempo."""

    def __init__(self, id=6879):
        self.data = get_forecast(id)

    def print(self):
        for day in self.data:
            for key in day:
                if type(day[key]) == dict:
                    for param in day[key]:
                        print("{:<30}\t{:<}".format(
                            param, day[key][param]))
                else:
                    print("{:<30}\t{:<}".format(
                        key, day[key]))
            print()

    def prec(self, dia=dt.datetime.now(), dur=dt.timedelta(days=1)):
        #soma = 0
        #for horario in self.data:
        #    hora = parse(horario["date"])
        #    if hora >= dia and hora <= dia + dur:
        #        soma += horario['rain']['precipitation']
        #return(soma)

        for horario in self.data:
            hora = parse(horario["date"])
            if hora >= dia and hora <= dia + dur and hora.hour == 23:
                return(horario['rain']['precipitation'])




# Tempo atual
def get_tempo(id=6879):
    """Faz a requisição do tempo."""

    cmd = "/api/v1/weather/locale/"\
        +str(id) + "/current?token=" + token
    r = requests.get(link + cmd).json()
    return(r['data'])


class tempo:
    """Classe das condições de tempo."""

    def __init__(self, id=6879):
        self.data = get_tempo(id)
        self.id = id

    def refresh(self):
        self.data = get_tempo(self.id)

    def change_id(self, id):
        self.id = id
        self.refresh()

    def print(self):
        for key in self.data:
            print("{:<30}\t{:<}".format(key, self.data[key]))




# Funções gerais
def get_id(cidade, estado):
    """Descobre id de uma cidade."""

    cidade.replace(" ", "+")
    cmd = "/api/v1/locale/city?name="\
        +cidade + "&state=" + estado +"&token=" + token
    
    r = requests.get(link + cmd).json()
    return(r[0]['id'])
