import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

# 1- Fuzzificação


# Antecedentes
tps = ctrl.Antecedent(np.arange(0, 100, 0.01), 'TPS')
rpm = ctrl.Antecedent(np.arange(0, 5000, 0.01), 'RPM')
speed = ctrl.Antecedent(np.arange(0, 60, 0.01), 'Speed')

# Saída
consumption = ctrl.Consequent(np.arange(0, 17, 0.01), 'Fuzzy_Fuel_Consumption')

# 2 - Inferência
tps['Very_Low'] = fuzz.trapmf(tps.universe, [0, 0, 3, 5])
tps['Low'] = fuzz.trapmf(tps.universe, [3, 5, 10, 40])
tps['Medium'] = fuzz.trimf(tps.universe, [20, 50, 80])
tps['High'] = fuzz.trapmf(tps.universe, [60, 90, 100, 100])

rpm['Very_Low'] = fuzz.trapmf(rpm.universe, [0, 0, 417, 833])
rpm['Low'] = fuzz.trapmf(rpm.universe, [417, 833, 1250, 1667])
rpm['Medium'] = fuzz.trapmf(rpm.universe, [1250, 1667, 2083, 2500])
rpm['High'] = fuzz.trapmf(rpm.universe, [2083, 2500, 2917, 3333])
rpm['Very_High'] = fuzz.trapmf(rpm.universe, [2917, 3333, 5000, 5000])

speed['Very_Low'] = fuzz.trapmf(speed.universe, [0, 0, 3, 5])
speed['Low'] = fuzz.trapmf(speed.universe, [3, 5, 10.35, 17.92])
speed['Medium'] = fuzz.trapmf(speed.universe, [10.35, 17.92, 23.89, 29.86])
speed['High'] = fuzz.trapmf(speed.universe, [23.89, 29.6, 35.83, 41.81])
speed['Very_High'] = fuzz.trapmf(speed.universe, [32.85, 44.80, 60, 60])

consumption['Very_Low'] = fuzz.trapmf(consumption.universe, [0, 0, 1.1, 2.21])
consumption['Low'] = fuzz.trapmf(consumption.universe, [1.1, 2.21, 3.32, 4.43])
consumption['Medium'] = fuzz.trapmf(consumption.universe, [3.3, 4.42, 5.53, 6.64])
consumption['High'] = fuzz.trapmf(consumption.universe, [5.53, 6.64, 8.85, 9.96])
consumption['Very_High'] = fuzz.trapmf(consumption.universe, [8.3, 9.96, 17, 17])

# Regras
rule1 = ctrl.Rule(rpm['Very_Low'], consumption['Very_Low'])
rule2 = ctrl.Rule(tps['Low'] & rpm['Low'] & (speed['Low'] | speed['Medium'] | speed['High']), consumption['Very_Low'])
rule3 = ctrl.Rule(tps['Low'] & rpm['Low'] & speed['Very_High'], consumption['Low'])
rule4 = ctrl.Rule(tps['Medium'] & rpm['Low'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Medium'])
rule5 = ctrl.Rule(tps['High'] & rpm['Low'] & (speed['Low'] | speed['Medium']), consumption['Low'])
rule6 = ctrl.Rule(tps['High'] & rpm['Low'] & speed['High'], consumption['Very_High'])
rule7 = ctrl.Rule(tps['High'] & rpm['Low'] & speed['Very_High'], consumption['High'])
rule8 = ctrl.Rule(tps['Low'] & rpm['Medium'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Low'])
rule9 = ctrl.Rule(tps['Medium'] & rpm['Medium'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Medium'])
rule10 = ctrl.Rule(tps['High'] & rpm['Medium'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Medium'])
rule11 = ctrl.Rule(tps['Low'] & rpm['High'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Low'])
rule12 = ctrl.Rule(tps['Medium'] & rpm['High'] & speed['Low'], consumption['Medium'])
rule13 = ctrl.Rule(tps['Medium'] & rpm['High'] & (speed['Medium'] | speed['High'] | speed['Very_High']), consumption['High'])
rule14 = ctrl.Rule(tps['High'] & rpm['High'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['High'])
rule15 = ctrl.Rule(tps['Low'] & rpm['Very_High'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Medium'])
rule16 = ctrl.Rule(tps['Medium'] & rpm['Very_High'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['High'])
rule17 = ctrl.Rule(tps['High'] & rpm['Very_High'] & (speed['Low'] | speed['Medium'] | speed['High'] | speed['Very_High']), consumption['Very_High'])
rule18 = ctrl.Rule(speed['Very_Low'], consumption['Very_Low'])
rule19 = ctrl.Rule(tps['Very_Low'], consumption['Very_Low'])

# 3 - Defuzzificação
rules = [
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19
]


result_ctrl = ctrl.ControlSystem(rules)
result_simulator = ctrl.ControlSystemSimulation(result_ctrl)

result_simulator.defuzzify_method = 'centroid'

# Ler dados do CSV
data = pd.read_csv('data.csv', skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], nrows=54000)

# Lista para armazenar os resultados de consumo
consumption_results = []

# Processar cada linha de dados
for index, row in data.iterrows():
    result_simulator.input['TPS'] = row['Throttle']
    result_simulator.input['RPM'] = row['Engine0_RPM']
    result_simulator.input['Speed'] = row['Speed']
    
    # Computar
    result_simulator.compute()
    
    # Armazenar o resultado
    consumption_results.append(result_simulator.output['Fuzzy_Fuel_Consumption'])

# Adicionar os resultados ao DataFrame
data['Fuzzy_Fuel_Consumption'] = consumption_results



# Plotar consumo de combustível real e fuzzy  Separados
'''
# Plotar o gráfico de consumo de combustível
fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
axs[0].plot(data.index, data['FuelUsePerHour'], label='Actual Fuel Consumption', color='blue')
axs[0].set_ylabel('Actual Fuel Consumption')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(data.index, data['Fuzzy_Fuel_Consumption'], label='Fuzzy Fuel Consumption', color='red')
axs[1].set_ylabel('Fuzzy Fuel Consumption')
axs[1].set_xlabel('Index')
axs[1].legend()
axs[1].grid(True)

plt.suptitle('Fuel Consumption Comparison')
plt.show()
'''
# Plotar o gráfico de consumo de combustível sobrepostos
plt.plot(data.index, data['FuelUsePerHour'], label='Actual Fuel Consumption')
plt.plot(data.index, data['Fuzzy_Fuel_Consumption'], label='Fuzzy Fuel Consumption')
plt.xlabel('Index')
plt.ylabel('Fuel Consumption')
plt.title('Comparison of Actual and Fuzzy Fuel Consumption')
plt.legend()
plt.grid(True)
plt.show()