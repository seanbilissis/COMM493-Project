from flask import Flask, render_template, url_for, redirect, request
import os
import Tables as tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import folium

#Set new plotting default font
plt.rcParams.update({'font.size': 8})

'''
Get most recent weeks Average Sentiment
'''
weekly_sent = round(tb.dfAverage.loc[tb.dfAverage.index[-1], "Average"],2)

'''
Generate the Sentiment and Sales Over Time graph
'''
#Plot sentiment data
t = np.arange(len(tb.dfSentimentandSales["Week"]))+2
fig, ax1 = plt.subplots()
ax1.bar(t, tb.dfSentimentandSales['Average'], color='#3a7ebf')
ax1.set_xlabel('Week')
ax1.set_ylabel('Average Sentiment')
ax1.tick_params(axis='y')
#Remove borders
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax2 = ax1.twinx()

#Plot sales data as overlapping line
color = '#98bde0'
ax2.set_ylabel('Sales')
ax2.plot(t, tb.dfSentimentandSales['Sales'], color=color)
ax2.tick_params(axis='y')
#Remove borders
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
plt.tight_layout()
#Save and close
plt.savefig('static/img/Sent_Sales.png')
plt.clf()
plt.cla()
plt.close()

'''
Generate Chrolopleth Map for Sentiment by Country
'''
#Create Choropleth Map (Folium)
country_geo = 'world-countries.json'
map_label = "Average Sentiment by Country"
bin_bounds = [-1, -0.67, -0.33, 0, 0.33, 0.67, 1]
map = folium.Map(location=[48,-102], tiles='cartodbpositron', zoom_start=3, min_zoom=2)
map.choropleth(geo_data=country_geo, data=tb.dfCountry,
            columns=['Country', 'Sentiment'], key_on='feature.id',
            bins=bin_bounds, fill_color='RdBu', nan_fill_color='lightgray', 
            fill_opacity=0.5, line_opacity=0.2,
            legend_name=map_label, highlight=True)
map.save('templates/choropleth.html')

'''
Generate Sentiment by Age Group graph
'''
#Plot data
agePlot = np.arange(len(tb.dfAges["Age Group"]))
plt.bar(agePlot, tb.dfAges["Average Sentiment"], color='#3a7ebf')
plt.xticks(agePlot, tb.dfAges["Age Group"])
plt.ylabel('Average Sentiment')
plt.tight_layout()
#Remove border
ax = plt.axes()
for spine in ax.spines.values():
    spine.set_visible(False)
#Save and close
plt.savefig('static/img/Sent_Age_Group.png')
plt.clf()
plt.cla()
plt.close()


''' 
Generate Sentiment by Gender graph
'''
#Plot data
genderPlot = np.arange(len(tb.dfGender["Gender"]))
plt.bar(genderPlot, tb.dfGender["Average Sentiment"], color='#3a7ebf')
plt.xticks(genderPlot, tb.dfGender["Gender"])
plt.ylabel('Average Sentiment')
plt.tight_layout()
#Remove borders
ax = plt.axes()
for spine in ax.spines.values():
    spine.set_visible(False)
#Save and close
plt.savefig('static/img/Sent_Gender.png')
plt.clf()
plt.cla()
plt.close()

'''
Generates Sentiment by Product Style graph
'''
#Plot data
prodPlot = np.arange(len(tb.dfProductStyle["Product Style"]))
plt.bar(prodPlot, tb.dfProductStyle["Average Sentiment"], color='#3a7ebf')
plt.xticks(prodPlot, tb.dfProductStyle["Product Style"], rotation=50, fontsize='8', horizontalalignment='right')
plt.ylabel('Average Sentiment')
plt.tight_layout()
#Remove borders
ax = plt.axes()
for spine in ax.spines.values():
    spine.set_visible(False)
#Save and close
plt.savefig('static/img/Sent_Product_Line.png')
plt.clf()
plt.cla()
plt.close()

'''
Create table for 10 worst sentiment changes by customer
'''


'''
Create and route Flask App
'''
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username']=='CEO' and request.form['password']=='CEO':
            return redirect(url_for('ceo_page'))
        elif request.form['username']=='CMO' and request.form['password']=='CMO':
            return redirect(url_for('cmo_page'))
        else: 
            error = 'Invalid Credentials'
    return render_template('login.html', error=error)

@app.route('/cmo')
def cmo_page():
    return render_template('cmo.html', weeklySent = weekly_sent,
        imgSentAge = 'static/img/Sent_Age_Group.png', 
        imgSentGen = 'static/img/Sent_Gender.png', 
        topSentDec = tb.dfAscending5.to_html(index=False, justify='center'))

@app.route('/ceo')
def ceo_page():
    return render_template('ceo.html', weeklySent = weekly_sent, 
        imgSentSales = 'static/img/Sent_Sales.png', imgSentProduct = 'static/img/Sent_Product_Line.png',
        topPraise = tb.dfPraise.to_html(index=False, justify='center'), 
        topComplaints = tb.dfComplaints.to_html(index=False, justify='center'))

@app.route('/choropleth')
def choropleth():
    return render_template('choropleth.html')

if __name__ == "__main__":
    app.run(debug=True)


