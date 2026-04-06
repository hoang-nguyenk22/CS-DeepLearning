from stackapi import StackAPI
import pandas as pd
import time
import os

def fetch_cs_data(target_count=5000000, csv_path = 'data/cs_raw.csv'):
    key = 'rl_oEDaHHZDZpBQYbecJxwqdr3Ss'
    SITE = StackAPI('cs', key=key)
    SITE.page_size = 100
    
    data = []
    page = 1
    total_fetched = 0
    
    os.makedirs('data', exist_ok=True)
    
    
    while total_fetched < target_count:
        try:
            questions = SITE.fetch('questions', 
                                  page=page, 
                                  min=1, 
                                  sort='votes', 
                                  filter='withbody')
            
            if 'items' not in questions or not questions['items']:
                break
                
            batch_data = []
            for item in questions['items']:
                batch_data.append({
                    'title': item.get('title'),
                    'body': item.get('body'),
                    'tags': '|'.join(item.get('tags', []))
                })
            
            data.extend(batch_data)
            total_fetched += len(batch_data)
            print(f"Progress: {total_fetched}/{target_count} (Page {page})")
            
            if page % 100 == 0:
                df_batch = pd.DataFrame(data)
                df_batch.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))
                data = []
                
            page += 1
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Stop at page {page}: {e}")
            break

    if data:
        pd.DataFrame(data).to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))
        return csv_path

#path = fetch_cs_data(75000)