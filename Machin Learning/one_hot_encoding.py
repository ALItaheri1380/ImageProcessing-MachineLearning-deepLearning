import geopy.distance
import pandas as pd;
from sklearn.linear_model import LinearRegression;
from geopy.geocoders import Nominatim
#********************Sample*******************************#
Reg = LinearRegression()
#********************GET LAT AND LONG*********************#


myState = input('Enter your province = ')

geolocator = Nominatim(user_agent="MyApp")

loc = input('Enter your location = ')

location_company = geolocator.geocode(myState)

location = geolocator.geocode(loc)

lat = location.latitude

long = location.longitude

lat_company = location_company.latitude
long_company = location_company.longitude
#******************GET DISTACE******************************#

coords_1 = (lat,long)
coords_2 = (lat_company,long_company)

dist = geopy.distance.distance(coords_1, coords_2).km
#*************************PREDICT***************************#

mylist = []
for i in range(32):
    mylist.append(0)
if(myState == "Guilan"):
    mylist[0] +=1
elif(myState == 'alborz'):
    mylist[1] +=1
elif(myState == 'ardebil'):
    mylist[2] +=1
elif(myState == 'azarbayejangharbi'):
    mylist[3] +=1
elif(myState == 'azarbayejansharghi'):
    mylist[4] +=1
elif(myState == 'booshehr'):
    mylist[5] +=1
elif(mylist == 'chaharmahal'):
    mylist[6] +=1
elif(myState == 'esfahan'):
    mylist[7] +=1
elif(myState == 'fars'):
    mylist[8] +=1
elif(myState == "ghazvin"):
    mylist[9] +=1
elif(myState == "ghom"):
    mylist[10] +=1
elif(myState == "golestan"):
    mylist[11] +=1
elif(myState == "hamedan"):
    mylist[12] +=1
elif(myState == "hormozgan"):
    mylist[13] +=1
elif(myState == "ilam"):
    mylist[14] +=1
elif(myState == "kerman"):
    mylist[15] +=1
elif(myState == "kermanshah"):
    mylist[16] +=1
elif(myState == "khoozestan"):
    mylist[17] +=1
elif(myState == "khorasanjonoobi"):
    mylist[18] +=1
elif(myState == "khorasanrazavi"):
    mylist[19] +=1
elif(myState == 'khorasanshomali'):
    mylist[20] +=1
elif(myState == 'kohgiloye'):
    mylist[21] +=1
elif(myState == 'kordestan'):
    mylist[22] +=1
elif(myState == 'lorestan'):
    mylist[23] +=1
elif(myState == 'markazi'):
    mylist[24] +=1
elif(myState == 'mazandaran'):
    mylist[25] +=1
elif(myState == 'semnan'):
    mylist[26] +=1
elif(myState == 'shiraz'):
    mylist[27] +=1
elif(myState == 'sistanvabaloochestan'):
    mylist[28] +=1
elif(myState == 'tehran'):
    mylist[29] +=1
elif(myState == 'yazd'):
    mylist[30] +=1
elif(myState == 'zanjan'):
    mylist[31] +=1                  
else:
    print('This state does not exist...!!!')
    exit()        
#****************************Predict****************************#                                                                                               
df = pd.read_csv('one_hot_encoding.csv')

df2 = pd.get_dummies(df.twon)

df3 = pd.concat([df,df2],axis='columns')

df4 = df3.drop(['twon'],axis = "columns")

all = df4.drop(['price'],axis='columns')

P = df4.price

Reg.fit(all,P)

print(Reg.predict
([[dist,mylist[0],mylist[1],mylist[2],mylist[3],mylist[4]
,mylist[5],mylist[6],mylist[7],mylist[8],mylist[9],mylist[10]
,mylist[11],mylist[12],mylist[13],mylist[14],mylist[15],mylist[16]
,mylist[17],mylist[18],mylist[19],mylist[20],mylist[21],mylist[22]
,mylist[23],mylist[24],mylist[25],mylist[26],mylist[27],mylist[28]
,mylist[29],mylist[30],mylist[31]
]]))