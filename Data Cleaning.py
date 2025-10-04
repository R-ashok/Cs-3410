import pandas 
 
data = pandas.read_excel("games.csv.xlsx")        
   
print(data.shape)      
print()     
print(data.columns)  
print() 
print(data.info())
print() 


print(data.dtypes)  
print() 
for column in data.columns: #report number of data types per column  
    print(f"{column}: {data[column].map(type).nunique()} types")     
print() 
    
print(data['white_id'].map(type).value_counts()) #print no. of enteries per data type   
print() 
print(data['black_id'].map(type).value_counts()) #print no. of enteries per data type
print()     
              

def NoLetters(x): #to clean player ids that have registered with invalid names (as per lichess format)
    if not isinstance(x, str):    
        return True           
    for ch in x:    
        if ch.isalpha():                
            return False                                   
    return True  

invalid_white = data["white_id"].apply(NoLetters)           
invalid_black = data["black_id"].apply(NoLetters) 
invalid_rows = invalid_white | invalid_black        
  
print(f"Found {invalid_rows.sum()} rows with no letters in player IDs.")
print()     
         
data = data[~invalid_rows].reset_index(drop=True)    

print(f"New dataset shape:{data.shape}")        
print()      
               

print(data['white_id'].map(type).value_counts()) #print no. of enteries per data type   
print() 
print(data['black_id'].map(type).value_counts()) #print no. of enteries per data type
print()                                       
    