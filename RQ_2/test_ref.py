from bson import ObjectId
from pymongo import MongoClient
from multiprocessing import Pool
from pymongo.cursor import Cursor
import time
import pickle



client = MongoClient("mongodb://localhost:27017/")
db = client['smartshark']
refactoring = db['refactoring']

def check_ref_commit(commit_id : str) -> bool:
    q = refactoring.find_one({'commit_id' : commit_id})
    print(q)
    if q is not None:
        return True
    return False


def pickle_if_slow(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if end_time - start_time > 2:
            with open(f"{func.__name__}output.pickle", "wb") as f:
                pickle.dump(result, f)
        pickle_msg = f"Function {func.__name__} took {end_time - start_time} seconds to complete.\n Pickled output to {func.__name__}output.pickle."
        print(pickle_msg)
        return result
    return wrapper

def cursor_to_list(cursor, num_processes: int = 4) -> list:
    """
    Convert pymongo cursor to list using parallel processing.
    """
    # Get the total number of documents in the cursor
    total_docs = sum(1 for _ in cursor.clone())
    print(total_docs)

    # Define a helper function to fetch documents from the cursor
    def fetch_documents(offset):
        return list(cursor.skip(offset).limit(batch_size))

    # Define batch size and offsets
    batch_size = 10000
    offsets = range(0, total_docs, batch_size)

    # Create a pool of worker processes
    pool = Pool(processes=num_processes)
    res = []
    for batch in offsets:
        r = pool.apply_async(fetch_documents, args=(batch,))
        res.append(r.get())
    pool.close()
    pool.join()


    # Flatten the list of results and return
    return res

@pickle_if_slow
def slow_function():
    time.sleep(3)
    return [1, 2, 3, 4, 5]

if __name__ == '__main__':
    #print(check_ref_commit(ObjectId('5bf51c47d2f8190d90f3c45')))

    from tqdm import tqdm

    # show a progress bar while the function is running
    for i in tqdm(range(10000000)):
        pass
    

    
   