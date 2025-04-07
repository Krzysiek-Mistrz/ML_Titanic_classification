## Titanic ML Prediction

Ten projekt zawiera kod do klasyfikacji pasażerów Titanica na podstawie danych z zestawu danych dostępnego w seaborn (dataset titanic). Wykorzystano m.in. techniki przetwarzania danych, pipeline oraz GridSearchCV do strojenia hiperparametrów modeli. W projekcie testowane są dwa podejścia: RandomForestClassifier oraz LogisticRegression.  
Zawartość  

    *Preprocessing*:  
    Rozdzielenie cech numerycznych i kategorycznych, uzupełnianie brakujących wartości oraz skalowanie/one-hot encoding.    

    *Model*:  
    Pipeline łączący preprocessing z modelem klasyfikacyjnym. Na początkowym etapie wykorzystywany jest RandomForestClassifier, a następnie testowany jest również LogisticRegression.  

    *Strojenie hiperparametrów*:  
    GridSearchCV wykorzystany do wyszukania optymalnych hiperparametrów dla modelu.  

    *Ewaluacja*:  
    Raport klasyfikacji, macierz pomyłek oraz wizualizacja ważności cech dla najlepszego modelu.  

    *Przykład predykcji dla użytkownika*:  
    Sekcja umożliwiająca wprowadzenie danych użytkownika i otrzymanie przewidywania (czy pasażer przeżył).  

## Instalacja

Upewnij się, że masz zainstalowane następujące biblioteki:  
    numpy  
    pandas  
    matplotlib  
    seaborn  
    scikit-learn  

Możesz zainstalować wymagane pakiety za pomocą pip:  
```pip install numpy pandas matplotlib seaborn scikit-learn```  
Uruchomienie  
Sklonuj repozytorium:  
```git clone https://github.com/Krzysiek-Mistrz/ML_Titanic_classification.git```  

Przejdź do katalogu projektu:  
`cd ML_Titanic_classification`  

Uruchom skrypt:  
    ```python classification_model.py```  

Struktura kodu:  
Kod wykonuje następujące kroki:  

    Wczytanie danych:  
    Wykorzystano dataset titanic z biblioteki seaborn.  
 
    Przygotowanie danych:  

        Wybrano cechy: pclass, sex, age, sibsp, parch, fare, class, who, adult_male, alone.

        Podzielono dane na zbiór treningowy i testowy z zachowaniem proporcji klas.  

    Preprocessing:  

        Dla cech numerycznych: imputacja medianą oraz standaryzacja.

        Dla cech kategorycznych: imputacja najczęstszą wartością oraz one-hot encoding.

        Wszystkie kroki łączone są przy pomocy ColumnTransformer.  

    Budowa pipeline:  
    Pipeline łączy preprocessing z modelem RandomForestClassifier, a następnie przy użyciu GridSearchCV optymalizowane są hiperparametry modelu.  

    Ewaluacja:  

        Obliczany jest raport klasyfikacji i wyświetlana macierz pomyłek.

        Wizualizacja ważności cech dla najlepszego modelu.

        Wypisywany jest wynik test set accuracy.

    Test modelu z LogisticRegression:  
    Pipeline jest modyfikowany do użycia LogisticRegression oraz ponownie oceniany.  