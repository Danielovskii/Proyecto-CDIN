import numpy as np
import pandas as pd
import string
import scipy.spatial.distance as sc

class CDIN():
    def __init__(self, df):
        self.data = df
        
    def get_cuantitativos(self):
        df_cuantitativos = self.data.select_dtypes(include = 'number')
        columns_cuantitativos = df_cuantitativos.columns
        return df_cuantitativos, columns_cuantitativos
    
    def get_cualitativos(self):
        df_cualitativos = self.data.select_dtypes(exclude = 'object')
        columns_cualitativos = df_cualitativos.columns
        return df_cualitativos, columns_cualitativos
    
    def get_binaries(self):
        df_bin = self.data.select_dtypes(include=['object']).copy()
        column_aux = df_bin.columns
        column_bin = []
        
        for col in column_aux:
            if df_bin[col].nunique() == 2:
                column_bin.append(col)
            
        return df_bin[column_bin], column_bin 
    
    @staticmethod
    def remove_whitespace(x):
        try:
            x = " ".join(x.split())
        except:
            pass
        return x
    
    @staticmethod
    def lower_text(x):
        try:
            x = x.lower()
        except:
            pass
        return x

    @staticmethod
    def upper_text(x):
        try:
            x = x.upper()
        except:
            pass
        return x

    @staticmethod
    def capitalize_text(x):
        try:
            palabras = []
            for r in x.split():
                palabras.append(r.capitalize())

            x = " ".join(palabras)
        except:
            pass
        return x
    
    @staticmethod
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation).strip()
        except:
            print(f'{x} no es una String')
        return x

    @staticmethod
    def remove_digits(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.digits).strip()
        except:
            print(f'{x} no es una String')
        return x
    
    @staticmethod
    def dqr(data):
        columnas = list(data.columns.values)
        columns = pd.DataFrame(columnas, columns = ['Nombre_Columnas'], index=columnas)
        dtypes = pd.DataFrame(data.dtypes, columns=['Tipo_Datos'])
        no_nulos = pd.DataFrame(data.count(), columns = ['Valores_Presentes'] )
        missing_values = pd.DataFrame(data.isnull().sum(), columns = ['Valores_Faltantes'])
        
        unique_values = pd.DataFrame(columns = ['Valores_Unicos'])
        for col in columnas:
            unique_values.loc[col] = [data[col].nunique()]
            
        # Columna Is Categorical
        categorical = pd.DataFrame(columns = ['Is_Categorical'], index=columnas)
        for col in columnas:
            if data[col].dtypes == 'object':
                categorical.loc[col] = [True]
            elif data[col].nunique() <= 5:
                categorical.loc[col] = ['Posible']
            else:
                categorical.loc[col] = ['No aplica']
                    
        
        # Aplicar el análisis estadístico solo a las columnas que son numericas    
        max_values = pd.DataFrame(columns = ['Max_Values'])
        for col in columnas:
            try:
                if categorical.loc[col][0] != True:
                    max_values.loc[col] = [data[col].max()]
            except:
                pass
            
        min_values = pd.DataFrame(columns = ['Min_Values'])
        for col in columnas:
            try:
                if categorical.loc[col][0] != True:
                    min_values.loc[col] = [data[col].min()]
            except:
                pass
        
        mean = pd.DataFrame(columns = ['Mean'])
        for col in columnas:
            try:
                if categorical.loc[col][0] != True:
                    mean.loc[col] = [round(data[col].mean(),3)]
            except:
                pass
            
        std = pd.DataFrame(columns = ['Std'])
        for col in columnas:
            try:
                if categorical.loc[col][0] != True:
                    std.loc[col] = [round(data[col].std(),3)]
            except:
                pass
            
        # Otra que muestre las categorias si son pocas
        categories = pd.DataFrame(columns = ['Categories'])
        for col in columnas:
            if categorical.loc[col][0] != 'No aplica' and data[col].nunique() <= 10:
                    categories.loc[col] = [data[col].unique()]
        
        
        return columns.join(dtypes).join(no_nulos).join(missing_values).join(unique_values).join(categorical).join(categories).join(max_values).join(min_values).join(mean).join(std)
    
    @staticmethod
    def categorize_columns(data):
        categorical_columns = []
        continuous_columns = []
        binary_columns = []
        ordinal_categorical_columns = []
        nominal_categorical_columns = []

        for j in data.columns:
            if data[j].dtype == 'object':
                if data[j].nunique() == 2:
                    binary_columns.append(j)
                elif data[j].nunique() == 16:
                    ordinal_categorical_columns.append(j)
                else:
                    nominal_categorical_columns.append(j)
            else:
                continuous_columns.append(j)

        print("Columnas Categóricas (Nominales):")
        print(nominal_categorical_columns)
        print("\nColumnas Categóricas (Ordinales):")
        print(ordinal_categorical_columns)
        print("\nColumnas Continuas:")
        print(continuous_columns)
        print("\nColumnas Binarias:")
        print(binary_columns)
    
    ## Emparejamiento
    
    def emp_simple(y_i, y_j) :
        mc = confusion_matrix(y_i, y_j)
        d = mc[0,0]
        c = mc[0,1]
        b = mc[1,0]
        a = mc[1,1]
        
        return (a+d)/(a+b+c+d)
    
    def jaccard(y_i, y_j) :
        mc = confusion_matrix(y_i, y_j)
        d = mc[0,0]
        c = mc[0,1]
        b = mc[1,0]
        a = mc[1,1]
        
        return (b+c)/(a+b+c)
    
    # Métodos para el cálculo de índices y distancias de similitud
    @staticmethod
    def pdistance_matrix(df, metric):
        '''
        Este método obtiene la matriz de distancias de df apartir de la métrica dada en el argumento de entrada
        df: DataFrame
        metric: Métrica (string)
        '''
        values = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
            'jaccard', 'jensenshannon', 'kulczynski1',
            'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            'sqeuclidean', 'yule']
        if metric in values:
            D = sc.squareform(sc.pdist(df.values,metric))
            return pd.DataFrame(D)
        else:
            print("Valor de métrica inválido")