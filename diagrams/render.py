
from os import environ
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(environ.get("RAM_DIAGRAM_OUTPUT", ROOT / "diagrams"))
FONT = Path(environ.get("RAM_DIAGRAM_FONTS", "/System/Library/Fonts/Supplemental"))

def font(size=36,bold=False):
    return ImageFont.truetype(str(FONT/("Times New Roman Bold.ttf" if bold else "Times New Roman.ttf")),size)

def canvas(h):
    im=Image.new('RGB',(1700,h),'white')
    return im,ImageDraw.Draw(im)

def line(d,pts,gray='#151515',width=4):
    d.line(pts,fill=gray,width=width,joint='curve')

def txt(d,where,s,size=37,bold=False,anchor='mm'):
    d.text(where,s,font=font(size,bold),fill='#111111',anchor=anchor)

def centered(d,x,y,lines,size=37,bold=False):
    spacing=int(size*1.17)
    for i,s in enumerate(lines):
        txt(d,(x,y+(i-(len(lines)-1)/2)*spacing),s,size,bold)

def arrow(d,pts,dashed=False):
    for a,b in zip(pts,pts[1:]):
        x,y=a;u,v=b
        length=((u-x)**2+(v-y)**2)**.5
        if not length: continue
        if dashed:
            for j in range(0,int(length),20):
                end=min(j+11,length)
                line(d,[(x+(u-x)*j/length,y+(v-y)*j/length),(x+(u-x)*end/length,y+(v-y)*end/length)])
        else: line(d,[a,b])
    x,y=pts[-1];u,v=pts[-2]
    le=((x-u)**2+(y-v)**2)**.5
    sx,sy=(x-u)/le,(y-v)/le
    d.polygon([(x,y),(x-20*sx-9*sy,y-20*sy+9*sx),(x-20*sx+9*sy,y-20*sy-9*sx)],fill='#151515')

def box(d,b,title,lines=(),kind='',gray='white',size=36):
    x1,y1,x2,y2=b
    d.rectangle(b,fill=gray,outline='#181818',width=3)
    xc=(x1+x2)//2
    total = 1+bool(kind)+len(lines)
    dy=size*1.22
    top=(y1+y2)/2-(total-1)*dy/2
    txt(d,(xc,top),title,size,True)
    pos=1
    if kind:
        txt(d,(xc,top+dy),kind,size-4)
        pos+=1
    for j,t in enumerate(lines,pos):txt(d,(xc,top+j*dy),t,size-2)

def save(im,name):
    TMP.mkdir(parents=True, exist_ok=True)
    p=TMP/(name+'.png')
    im.save(p,dpi=(254,254))
    return p

def histogram():
    with (ROOT/'reports/eda_target_7d.csv').open(encoding='utf-8-sig') as f:
        prices=[int(r['target_price_rub']) for r in csv.DictReader(f)]
    im,d=canvas(680)
    l,r,t,b=132,1640,65,555
    txt(d,(l,25),'Число наблюдений',36,anchor='lm')
    span=(r-l)/9
    for i in (0,30,60,90,120,150):
        y=b-(b-t)*i/150
        line(d,[(l,y),(r,y)],'#cccccc' if i else '#222222',2)
        txt(d,(l-24,y),str(i),31,anchor='rm')
    line(d,[(l,t),(l,b)],width=3)
    for j in range(9):
        n=sum(10000*j<=p<10000*(j+1) for p in prices)
        x=l+span*j
        height=(b-t)*n/150
        if n:
            d.rectangle((x+2,b-height,x+span-2,b-2),fill='#646464',outline='#151515',width=2)
            txt(d,(x+span/2,b-height-22),str(n),31)
    for j in range(10):
        x=l+span*j
        line(d,[(x,b),(x,b+11)],width=2)
        txt(d,(x,b+37),str(j*10),32)
    txt(d,((l+r)/2,652),'Цена через 7 дней, тыс. руб.',37)
    return save(im,'target')

def context():
    im,d=canvas(525)
    box(d,(50,15,650,145),'Покупатель / специалист по закупкам',('Смотрит прогноз и динамику цен',),'[Пользователь]',gray='#f4f4f4',size=32)
    box(d,(1050,15,1650,145),'Оператор',('Принимает порции и проверяет ошибки',),'[Пользователь]',gray='#f4f4f4',size=32)
    box(d,(475,260,1225,420),'Система анализа цен DDR5',('Публикует прогноз на 7 дней и показатели',),'[Программная система]',gray='#ececec',size=35)
    box(d,(36,265,342,415),'PCstonks',('История DNS',),'[Внешний источник]',size=32)
    arrow(d,[(350,150),(350,222),(630,222),(630,256)])
    arrow(d,[(1350,150),(1350,222),(1070,222),(1070,256)])
    txt(d,(480,184),'Смотрит прогноз',32)
    txt(d,(1225,184),'Загружает порцию',32)
    arrow(d,[(347,340),(468,340)])
    txt(d,(405,315),'Цены',29)
    txt(d,(220,473),'PCstonks предоставляет историю цен и наличия',32,anchor='lm')
    return save(im,'c4_context')

def actor(d,x,y,label):
    d.ellipse((x-25,y,x+25,y+50),outline='#151515',width=4)
    line(d,[(x,y+50),(x,y+114)])
    line(d,[(x-42,y+70),(x+42,y+70)])
    line(d,[(x-36,y+153),(x,y+114),(x+36,y+153)])
    centered(d,x,y+196,label,34,True)

def use_cases():
    im,d=canvas(642)
    d.rectangle((340,18,1360,612),outline='#171717',width=4)
    txt(d,(850,54),'Система анализа цен DDR5',39,True)
    actor(d,150,204,['Оператор'])
    actor(d,1550,199,['Покупатель /','закупщик'])
    cases=[(854,137,'Загрузить порцию'),(854,251,'Посмотреть протокол проверки'),(854,365,'Получить прогноз по артикулу'),(854,479,'Посмотреть показатели и динамику')]
    for x,y,s in cases:
        d.ellipse((x-360,y-45,x+360,y+45),outline='#171717',width=3)
        txt(d,(x,y),s,34)
    for y in (137,251):
        line(d,[(191,274),(489,y)])
    for y in (365,479):
        line(d,[(1508,269),(1219,y)])
    return save(im,'uml_use_case')

def sequence():
    im,d=canvas(1040)
    xs=[165,482,816,1150,1530]
    labels=[('Оператор',()),('Приём и контроль',('Пакетное приложение',)),('Признаки / ML',('Пакетное приложение',)),('Хранилище',('SQLite',)),('API',('FastAPI',))]
    for x,(name,sub) in zip(xs,labels):
        box(d,(x-147,15,x+147,107),name,sub,size=29)
        for y in range(108,1020,19):
            line(d,[(x,y),(x,y+11)],'#777777',2)
    def msg(i,j,y,t,dash=False):
        x1,x2=xs[i],xs[j]
        arrow(d,[(x1,y),(x2,y)],dash)
        txt(d,((x1+x2)/2,y-24),t,27)
    msg(0,1,170,'Загрузить CSV')
    msg(1,3,232,'Проверить версию, ключи и хеш')
    msg(3,1,282,'Состояние версии',True)
    d.rectangle((33,325,1672,974),outline='#333333',width=3)
    d.rectangle((33,325,122,371),fill='white',outline='#333333',width=3)
    txt(d,(76,346),'alt',32,True)
    txt(d,(150,349),'[Порция отклонена]',31,anchor='lm')
    msg(1,0,430,'Отчёт об ошибках',True)
    for x in range(35,1667,22):line(d,[(x,475),(x+11,475)],'#333333',2)
    txt(d,(60,500),'[Порция принята]',31,anchor='lm')
    msg(1,2,570,'Принятые строки')
    msg(2,3,630,'Читать историю и версию¹')
    msg(3,2,683,'История и версия¹',True)
    msg(2,3,747,'Записать выпуск')
    msg(3,2,801,'Подтверждение транзакции',True)
    msg(4,3,863,'Читать витрину')
    msg(3,4,915,'Опубликованная версия',True)
    txt(d,(45,1008),'¹ Версия модели берётся из SQLite, файл модели — из каталога артефактов.',28,anchor='lm')
    return save(im,'uml_sequence')

def lifecycle():
    im, d = canvas(600)
    positions = [(20, 25, 500, 155), (610, 25, 1090, 155), (1200, 25, 1680, 155),
                 (1200, 345, 1680, 475), (610, 345, 1090, 475), (20, 345, 500, 475)]
    labels = [
        ("1. Постановка задачи", ("Цена DDR5 на t + 7",)),
        ("2. Поставка данных", ("События PCstonks / DNS",)),
        ("3. Обучение и оценка", ("Временные выборки, MAE",)),
        ("4. Пакетный выпуск", ("Прогнозы и показатели SQLite",)),
        ("5. Сверка и мониторинг", ("Факты, MAE и покрытие",)),
        ("6. Решение о смене модели", ("Проверка источника и претендента",)),
    ]
    for bounds, (name, description) in zip(positions, labels):
        box(d, bounds, name, description, size=34)
    arrow(d, [(505, 90), (603, 90)])
    arrow(d, [(1095, 90), (1193, 90)])
    arrow(d, [(1440, 160), (1440, 339)])
    txt(d, (1430, 246), "Модель принята", 32, anchor="rm")
    arrow(d, [(1194, 410), (1095, 410)])
    arrow(d, [(603, 410), (505, 410)])
    arrow(d, [(260, 340), (260, 232), (1224, 232), (1224, 160)])
    txt(d, (735, 207), "Повторное обучение при подтверждённом ухудшении", 32)
    arrow(d, [(850, 480), (850, 550), (1440, 550), (1440, 481)])
    txt(d, (1145, 577), "Действующая версия при стабильном качестве", 32)
    return save(im, "lifecycle")


def structure():
    im, d = canvas(460)
    items = [
        ((20, 24, 500, 168), "Загрузка суточной порции", "События цен и наличия"),
        ((610, 24, 1090, 168), "Проверка и признаки", "Однозначные наблюдения"),
        ((1200, 24, 1680, 168), "Пакетный расчёт", "Прогнозы на t + 7 и KPI"),
        ((1200, 270, 1680, 414), "Витрина SQLite", "Артикулы, даты и версия"),
        ((610, 270, 1090, 414), "Панель, API, экспорт", "Чтение опубликованных данных"),
        ((20, 270, 500, 414), "Покупатель", "Бюджет и срок покупки"),
    ]
    for bounds, name, description in items:
        box(d, bounds, name, (description,), size=34)
    for start, finish in [((507, 96),(604, 96)),((1097, 96),(1193, 96)),
                          ((1440, 174),(1440, 263)),((1194, 342),(1097, 342)),
                          ((604, 342),(507, 342))]:
        arrow(d, [start, finish])
    txt(d, (1423, 220), "Единая транзакция", 31, anchor="rm")
    return save(im, "structure")


def scatter():
    im, d = canvas(695)
    l, t, r, b = 150, 68, 1610, 579
    txt(d, (l, 26), 'Цена через 7 дней, тыс. руб.', 36, anchor='lm')
    for tick in range(0, 91, 15):
        x = l + tick*(r-l)/90
        y = b - tick*(b-t)/90
        line(d, [(x,t),(x,b)], '#e1e1e1', 2)
        line(d, [(l,y),(r,y)], '#e1e1e1', 2)
        txt(d, (x,b+32), str(tick), 31)
        txt(d, (l-20,y), str(tick), 31, anchor='rm')
    line(d, [(l,t),(l,b),(r,b)], width=3)
    line(d, [(l,b),(r,t)], '#7d7d7d', 3)
    with (ROOT/'reports/eda_target_7d.csv').open(encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            x = l + int(row['current_price_rub'])*(r-l)/90000
            y = b - int(row['target_price_rub'])*(b-t)/90000
            d.ellipse((x-5,y-5,x+5,y+5), outline='#242424', width=2)
    txt(d, ((l+r)/2, 653), 'Текущая цена, тыс. руб.', 36)
    line(d, [(190,116),(244,116)], '#7d7d7d', 3)
    txt(d, (261,116), 'Цена без изменения', 30, anchor='lm')
    txt(d, (203,160), '○', 33)
    txt(d, (261,160), 'Наблюдаемая пара цен', 30, anchor='lm')
    return save(im, 'price_scatter')


def c4_containers():
    im,d=canvas(835)
    d.rectangle((230,148,1670,786),outline='#333333',width=4)
    txt(d,(249,179),'Система анализа цен DDR5',32,True,anchor='lm')
    box(d,(273,235,720,402),'Пакетное приложение',('Приём, признаки, прогноз, KPI',),'[Python, scikit-learn]',size=32)
    box(d,(816,235,1175,402),'Панель',('Фильтры и графики',),'[Streamlit]',size=32)
    box(d,(1276,235,1626,402),'API',('Прогнозы и показатели',),'[FastAPI]',size=32)
    box(d,(252,574,618,722),'Архив',('Снимки и суточные порции',),'[JSON, CSV]',size=29)
    box(d,(646,574,1015,722),'Артефакты модели',('Версии и метрики',),'[MLflow, файлы]',size=29)
    box(d,(1070,574,1611,722),'История и витрина',('Цены, прогнозы, KPI',),'[SQLite]',size=31)
    box(d,(288,14,710,115),'Оператор',('[Пользователь]',),size=32)
    box(d,(865,14,1329,115),'Покупатель / закупщик',('[Пользователь]',),size=32)
    box(d,(8,254,197,350),'PCstonks',('[Внешняя система]',),size=28)
    arrow(d,[(495,118),(495,229)])
    txt(d,(508,186),'CLI',29,anchor='lm')
    arrow(d,[(995,118),(995,229)])
    txt(d,(1010,186),'HTTP',29,anchor='lm')
    arrow(d,[(203,299),(268,299)])
    txt(d,(151,380),'HTTPS / JSON',27)
    arrow(d,[(400,406),(400,569)])
    txt(d,(415,495),'Файлы',28,anchor='lm')
    arrow(d,[(630,406),(630,481),(785,481),(785,570)])
    txt(d,(799,525),'Файлы модели',28,anchor='lm')
    arrow(d,[(684,407),(684,430),(1112,430),(1112,569)])
    txt(d,(874,450),'SQL, запись',28)
    arrow(d,[(1180,320),(1270,320)])
    txt(d,(1226,284),'HTTP',26)
    txt(d,(1226,348),'JSON',26)
    arrow(d,[(1438,407),(1438,569)])
    txt(d,(1451,490),'SQL, чтение',28,anchor='lm')
    return save(im,'c4_containers')


def activity():
    im,d=canvas(1115)
    def node(bounds, label, size=33):
        d.rounded_rectangle(bounds,radius=18,fill='white',outline='#161616',width=3)
        x,y,u,v=bounds
        txt(d,((x+u)/2,(y+v)/2),label,size)
    def diamond(x,y):
        d.polygon([(x-39,y),(x,y-28),(x+39,y),(x,y+28)],outline='#121212',fill='white',width=3)
    d.ellipse((831,13,869,51),fill='#111111')
    node((565,80,1135,158),'Прочитать CSV и манифест')
    node((565,194,1135,272),'Проверить схему и хеш')
    diamond(850,329)
    node((1120,289,1685,369),'Зафиксировать отказ')
    node((565,414,1135,492),'Очистить строки и проверить цены')
    node((565,528,1135,606),'Сформировать признаки на дату t')
    diamond(850,665)
    node((20,725,710,805),'Присоединить метки t + 7 и разделить по времени',30)
    node((20,840,710,920),'Настроить преобразования на train',31)
    node((990,725,1680,805),'Загрузить сохранённые параметры',31)
    diamond(850,938)
    node((550,990,1150,1068),'Преобразовать и сохранить признаки')
    for y, v in [(55,75),(163,189),(277,297),(362,409),(497,523),(611,632),(970,985)]:
        arrow(d,[(850,y),(850,v)])
    arrow(d,[(895,329),(1115,329)])
    txt(d,(1005,310),'[ошибка]',29)
    txt(d,(835,383),'[проверки пройдены]',29,anchor='rm')
    arrow(d,[(806,665),(365,665),(365,720)])
    arrow(d,[(894,665),(1335,665),(1335,720)])
    txt(d,(539,643),'[обучение]',29)
    txt(d,(1174,643),'[новая порция]',29)
    arrow(d,[(365,809),(365,835)])
    arrow(d,[(365,925),(365,938),(807,938)])
    arrow(d,[(1335,810),(1335,938),(894,938)])
    for x,y,y0 in [(1402,410,374),(850,1096,1073)]:
        d.ellipse((x-15,y-15,x+15,y+15),outline='#111111',width=3)
        d.ellipse((x-9,y-9,x+9,y+9),fill='#111111')
        arrow(d,[(x,y0),(x,y-17)])
    return save(im,'uml_activity')


if __name__ == "__main__":
    for figure in (lifecycle, structure, histogram, scatter, context, c4_containers,
                   use_cases, sequence, activity):
        print(figure())
