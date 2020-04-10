'''
Name of tables:
    dfAverage - This dataframe tells the average sentiment per week.
    dfSentimentandSales- This dataframe tells the average sentiment per week and sales per week.
    dfCountry - This dataframe provides Sentiment by country
    dfCustomers - This provides a list of customers with their sentiment not sorted (dfCustomerSortedLeastToMost and dfCustomerSortedMostToLeast are sorted)
    dfAges - Sentiment by age group
    dfGender - Sentiment by gender
    dfProduct - Sentiment for each product
    dfProductType - This provides an overview of what sentiment is across different product types (hat, sweater, shirt....)
    dfProductStyle - This provides an overview of what sentiment is across different product types (funny and bold, party prices, contemporary pieces)
    dfCustomerSentimentChange- This provides information on what customer's sentiment has changed the most over time

     NOTE: The below tables haven't been made cause im waiting on the labels. But I have provided something that shows what they should look like. 
           Please note: there could be more rows for these  
     dfComplaints - Shows complaints and their frequency
     dfPraise - shows praises and their frequency   
'''

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
import folium


#Importing Dataframes
countries=pd.read_csv("Countries.csv") 
customers=pd.read_csv("Customer.csv") 
geographies=pd.read_csv("Geographies.csv") 
orders=pd.read_csv("Orders.csv") 
products=pd.read_csv("Products.csv")
sales=pd.read_csv("Sales.csv")  
feedback=pd.read_csv("Analysis.csv") 
orderItems=pd.read_csv("Order_Items.csv")

'''
dfAverage
This dataframe tells the average sentiment per week.
Because it provides a more higher-level overview it is better suited for the CEO
'''
#Average Sentiment and Sales over time
latestComment=feedback["Week Posted"].max()
earliestComment=feedback["Week Posted"].min()

weeks=range(earliestComment,latestComment+1)
averages=[]
for item in weeks:
    count=0
    totalSum=0
    for index,row in feedback.iterrows():
        if row["Week Posted"]==item:
            count=count+1
            totalSum=totalSum+row["Sentiment"]
    average=totalSum/count
    averages.append(average)

dfData={"Week":weeks,"Average":averages}
dfAverage=pd.DataFrame(data=dfData)
            



'''
dfSentimentandSales
This dataframe tells the average sentiment per week and sales per week.
Because it provides a more higher-level overview it is better suited for the CEO
'''
dfData={"Week":weeks,"Average":averages,}
dfSentimentandSales=pd.DataFrame(data=dfData)
dfSentimentandSales["Sales"]=sales["Net Sales"][earliestComment-2:latestComment-1]

#Generates the Sentiment and Sales Graph
t = np.arange(len(dfSentimentandSales["Week"]))+2
fig, ax1 = plt.subplots()

ax1.bar(t, dfSentimentandSales['Average'], color='#3a7ebf')
ax1.set_xlabel('Week')
ax1.set_ylabel('Average Sentiment')
ax1.tick_params(axis='y')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax2 = ax1.twinx()

color = '#98bde0'
ax2.set_ylabel('Sales')
ax2.plot(t, dfSentimentandSales['Sales'], color=color)
ax2.tick_params(axis='y')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
plt.tight_layout()

plt.savefig('static/img/Sent_Sales.png')
plt.clf()
plt.cla()
plt.close()

'''
dfCountry
This dataframe provides Sentiment by country
'''
#different than Saqib's (new variables)

orderNumbers=feedback["Order_Number"].tolist()
Alpha3=[]
Sentiment=[]
Count=[]
Average=[]


for item in orderNumbers:
    
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    countryID=customers.loc[customers['Customer ID']==customerID,'Country'].iloc[0]
    countryAlpha3=countries.loc[countries['Text Code']==countryID,'Alpha-3 code'].iloc[0]
    
    if countryAlpha3 not in Alpha3:
        Alpha3.append(countryAlpha3)
        Sentiment.append(sentiment)
        Count.append(1)
    elif countryAlpha3 in Alpha3:
        CountryIndex=Alpha3.index(countryAlpha3)
        Sentiment[CountryIndex]=Sentiment[CountryIndex]+sentiment
        Count[CountryIndex]=Count[CountryIndex]+1

for item in Alpha3:
    CountryIndex=Alpha3.index(item)
    Average.append(Sentiment[CountryIndex]/Count[CountryIndex])

country_data={"Country":Alpha3, "Sentiment":Average}
dfCountry=pd.DataFrame(data=country_data)
    
#Create Choropleth Map (Folium)
country_geo = 'world-countries.json'
map_label = "Average Sentiment by Country"
bin_bounds = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
map = folium.Map(location=[48,-102], tiles='cartodbpositron', zoom_start=3, min_zoom=2)
map.choropleth(geo_data=country_geo, data=dfCountry,
            columns=['Country', 'Sentiment'], key_on='feature.id',
            bins=bin_bounds, fill_color='RdBu', nan_fill_color='lightgray', 
            fill_opacity=0.5, line_opacity=0.2,
            legend_name=map_label, highlight=True)
map.save('templates/choropleth.html')

    
'''
dfCustomers
This provides a list of customers with their sentiment sorted from highest to least
This is a more lower level view of what customers are feeling and is better suited for the marketing manager
'''    
#Sentiment by Customer (Most and Least) 
Customers=[]
Sentiment=[]
Count=[]
Average=[]
for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    
    if customerID not in Customers:
        Customers.append(customerID)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        CustomerIndex=Customers.index(customerID)
        Sentiment[CustomerIndex]=Sentiment[CustomerIndex]+sentiment
        Count[CustomerIndex]=Count[CustomerIndex]+1
for item in Customers:
    CustomerIndex=Customers.index(customerID)
    Average.append(Sentiment[CustomerIndex]/Count[CustomerIndex])

dfCustomers=pd.DataFrame(data={"Customer":Customers,"Average Sentiment":Sentiment})
dfCustomerSortedLeastToMost=dfCustomers.sort_values(by="Average Sentiment",ascending=True)
dfCustomerSortedMostToLeast=dfCustomers.sort_values(by="Average Sentiment",ascending=False)



'''
dfAges
this provides a dataframe of sentiment across various age groups. Suited for a histogram
Because of its more demographic view, it is better suited for the marketing dashboard
'''
Ages=[]
Sentiment=[]
Count=[]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    DOB=customers.loc[customers["Customer ID"]==customerID,"Birthday"].iloc[0]
    last2=DOB[-2:]
    last2=str(last2)
    
    if last2[0] == "0":
        year="20"+last2
        year=int(year)
    
    else:
        year="19"+last2
        year=int(year)
    age=2020-year
    
    if age not in Ages:
        Ages.append(age)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        AgeIndex=Ages.index(age)
        Sentiment[AgeIndex]=Sentiment[AgeIndex]+sentiment
        Count[AgeIndex]=Count[AgeIndex]+1
    
for item in Ages:
    AgeIndex=Ages.index(item)
    Average.append(Sentiment[AgeIndex]/Count[AgeIndex])

minAgeAc=min(Ages)
maxAgeAc=max(Ages)

if min(Ages)%5 != 0:
    minAge=min(Ages)-min(Ages)%5
else:
    minAge=min(Ages)

if max(Ages)%5 != 0:
    maxAge=(5-max(Ages)%5)+max(Ages)+5
else:
    maxAge=max(Ages)+5


bins=list(range(minAge,maxAge,5))



dfAges=pd.DataFrame(data={"Age":Ages,"Average_Sentiment":Sentiment,"Count":Count})
dfAges["Binned"]=pd.cut(dfAges["Age"],bins=bins)
dfAges["Bin Sentiment"]=dfAges.Binned.map(dfAges.groupby(["Binned"]).Average_Sentiment.mean())
dfAges=dfAges[["Binned","Bin Sentiment"]]
dfAges=dfAges.drop_duplicates()
dfAges=dfAges.rename(columns={"Binned":"Age Group","Bin Sentiment":"Average Sentiment"})


#Generates the Sent by Age Graph
agePlot = np.arange(len(dfAges["Age Group"]))
plt.bar(agePlot, dfAges["Average Sentiment"], color='#3a7ebf')
plt.xticks(agePlot, dfAges["Age Group"])
plt.ylabel('Average Sentiment')
plt.tight_layout()

ax = plt.axes()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig('static/img/Sent_Age_Group.png')
plt.clf()
plt.cla()
plt.close()
    
'''
dfGender
this provides a dataframe of sentiment across gender groups. 
Because of its more demographic view, it is better suited for the marketing dashboard
'''
Genders=[]
Sentiment=[]
Count=[]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    gender=customers.loc[customers["Customer ID"]==customerID,'Gender'].iloc[0]
    if gender==0:
        gender="Female"
    else:
        gender="Male"
    
    if gender not in Genders:
        Genders.append(gender)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        GenderIndex=Genders.index(gender)
        Sentiment[GenderIndex]=Sentiment[GenderIndex]+sentiment
        Count[GenderIndex]=Count[GenderIndex]+1

for item in Genders:
    GenderIndex=Genders.index(item)
    Average.append(Sentiment[GenderIndex]/Count[GenderIndex])
dfGender=pd.DataFrame(data={"Gender":Genders,"Average Sentiment":Average})

#Generates Sent by Gender Graph
genderPlot = np.arange(len(dfGender["Gender"]))
plt.bar(genderPlot, dfGender["Average Sentiment"], color='#3a7ebf')
plt.xticks(genderPlot, dfGender["Gender"])
plt.ylabel('Average Sentiment')

plt.tight_layout()
ax = plt.axes()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig('static/img/Sent_Gender.png')
plt.clf()
plt.cla()
plt.close()

'''
dfProduct
This provides an overview of what sentiment is across all products
This information is more lower level and is better suited for the CMO
'''
#Sentiment by Product
Products=[]
Sentiment=[]
Count=[]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    productID=orderItems.loc[orderItems["Order Number"]==item,"Product_ID"].iloc[0]
    
    if productID not in Products:
        Products.append(productID)
        Sentiment.append(sentiment)
        Count.append(1)
    else:
        productIndex=Products.index(productID)
        Sentiment[productIndex]=Sentiment[productIndex]+sentiment
        Count[productIndex]=Count[productIndex]+1
for item in Products:
    productIndex=Products.index(item)
    Average.append( Sentiment[productIndex]/ Count[productIndex])

dfProduct=pd.DataFrame(data={"Products":Products,"Average Sentiment":Average})

   
    

'''
dfProductType
This provides an overview of what sentiment is across different product types (hat, sweater, shirt....)
This information is more lower level and think it is better suited for the CMO
'''
ProductTypes=["hat", "sweater", "shirt", "jacket", "pant", "shoe", "blouse"]
Sentiment=[0,0,0,0,0,0,0]
Count=[0,0,0,0,0,0,0]
Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    productID=orderItems.loc[orderItems["Order Number"]==item,"Product_ID"].iloc[0]
    des=products.loc[products["Product_ID"]==productID,"Description"].iloc[0]
    des=des.lower()
    
    for p in ProductTypes:
        if p in des:
            index=ProductTypes.index(p)
            Count[index]=Count[index]+1
            Sentiment[index]=Sentiment[index]+sentiment


for item in ProductTypes:
    index=ProductTypes.index(item)
    if Count[index]!=0:
        index=ProductTypes.index(item)
        Average.append(Sentiment[index]/Count[index])
    else:
        Average.append(0)

dfProductType=pd.DataFrame(data={"Product Type":ProductTypes,"Average Sentiment":Average})
dfProductType = dfProductType[~(dfProductType == 0).any(axis=1)]            
            
    
    

'''
dfProductStyle
This provides an overview of what sentiment is across different product types (funny and bold, party prices, contemporary pieces)
This information is more lower level and think it is better suited for the CMO
'''
#Sentiment by Product Style (funky and bold, party peices, etc)
ProductStyle=["funky and bold", "party pieces", "contemporary and clean", "hipster trend", "dress to impress", "runway inspired", "romantic and bohemian","athletic","utility and basics","relaxed and casual"]

Sentiment=[0,0,0,0,0,0,0,0,0,0]
Count=[0,0,0,0,0,0,0,0,0,0]

Average=[]

for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    productID=orderItems.loc[orderItems["Order Number"]==item,"Product_ID"].iloc[0]
    des=products.loc[products["Product_ID"]==productID,"Description"].iloc[0]
    des=des.lower()
    
    for p in ProductStyle:
        if p in des:
            index=ProductStyle.index(p)
            Count[index]=Count[index]+1
            Sentiment[index]=Sentiment[index]+sentiment


for item in ProductStyle:
    index=ProductStyle.index(item)
    if Count[index]!=0:
        index=ProductStyle.index(item)
        Average.append(Sentiment[index]/Count[index])
    else:
        Average.append(0)

dfProductStyle=pd.DataFrame(data={"Product Style":ProductStyle,"Average Sentiment":Average})
dfProductStyle= dfProductStyle[~(dfProductStyle == 0).any(axis=1)]            


#Generates Sent by Product Line Graph
prodPlot = np.arange(len(dfProductStyle["Product Style"]))
plt.bar(prodPlot, dfProductStyle["Average Sentiment"], color='#3a7ebf')
plt.xticks(prodPlot, dfProductStyle["Product Style"], rotation=50, fontsize='8', horizontalalignment='right')
plt.ylabel('Average Sentiment')
#plt.title('Sentiment by Product Line')

#Styling
plt.tight_layout()
ax = plt.axes()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig('static/img/Sent_Product_Line.png')
plt.clf()
plt.cla()
plt.close()     


'''
dfCustomerSentimentChange
This provides information on what customer's sentiment has changed the most over time
This is lower level data and better suited for the CMO
'''
#Change in sentiment over time per customer
FirstandLast={}
for item in orderNumbers:
    sentiment=feedback.loc[feedback["Order_Number"]==item,"Sentiment"].iloc[0]
    customerID=orders.loc[orders['Order Number'] == item, 'Customer_ID'].iloc[0]
    keys=list(FirstandLast.keys())
    if customerID not in keys:
        FirstandLast[customerID]=[sentiment,0]
    else:
        FirstandLast[customerID][1]=sentiment

customerIDs=list(FirstandLast.keys())
sentimentChange=[]
for item in customerIDs:
    if len(FirstandLast[item])>1:
        first=FirstandLast[item][0]
        last=FirstandLast[item][1]
        change=last-first
        sentimentChange.append(change)


dfCustomerSentimentChange=pd.DataFrame(data={"CustomerID":customerIDs,"Change":sentimentChange})

#Gets df of 10 worst for dashboard  (need to include)
dfAscending10 = dfCustomerSentimentChange.nsmallest(10, "Change")

'''
dfComplaints - Shows complaints and their frequency
'''
complaints=["poorly priced","poor design","poor sizing"]
counts=[1,2,3]
dfComplaints=pd.DataFrame(data={"Complaint":complaints,"Count":counts})



'''
dfPraise - shows praises and their frequency
'''
praises=["well priced","well design","excellent sizing"]
counts=[1,2,3]
dfPraises=pd.DataFrame(data={"Praise":praises,"Count":counts})