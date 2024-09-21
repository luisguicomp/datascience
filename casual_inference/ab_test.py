# Dados simulados
import pandas as pd
import numpy as np
from dowhy import CausalModel

# Supondo que 'grupo' seja 1 para novo layout (B) e 0 para layout antigo (A)
data = pd.DataFrame({
    'grupo': np.random.choice([0, 1], size=100),  # Grupo A/B
    'vendas': np.random.normal(100, 10, size=100),  # Vendas simuladas
    'tempo_navegacao': np.random.normal(5, 1, size=100)  # Variável de confusão: tempo na página
})

# Definindo o modelo causal
model = CausalModel(
    data=data,
    treatment='grupo',
    outcome='vendas',
    common_causes=['tempo_navegacao']
)

# Identificando e estimando o efeito causal
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

# Resultado
print("Efeito causal estimado:", estimate.value)
