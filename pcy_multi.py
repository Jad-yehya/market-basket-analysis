from itertools import combinations
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import pickle as pkl

def mapper(args):
    """
    Map phase: Count candidate pairs in each chunk

    Parameters
    ----------
    args : tuple
        Tuple containing the chunk, frequent single items, and hash table size

    Returns
    -------
    candidate_counts : dict
        Dictionary containing the candidate pairs and their counts
    hash_table : numpy.ndarray
        Hash table containing the counts of candidate pairs
    """
    chunk, frequent_single_items, hash_table_size = args
    candidate_counts = {}
    hash_table = np.zeros((hash_table_size,), dtype=int)

    for transaction in chunk:
        items = set(transaction)

        # Generate candidate pairs
        pairs = combinations(items, 2)
        candidate_pairs = [(item1, item2) for item1, item2 in pairs if item1 in frequent_single_items and item2 in frequent_single_items]

        # Count candidate pairs in the hash table
        for pair in candidate_pairs:
            hash_value = hash(pair) % hash_table_size
            hash_table[hash_value] += 1
            candidate_counts[pair] = candidate_counts.get(pair, 0) + 1

    return candidate_counts, hash_table

def reducer(mapped_data, min_support, hash_table_size):
    """
    Reduce phase: Filter frequent itemsets based on minimum support and hash table counts
    
    Parameters
    ----------
    mapped_data : list
        List containing the mapped data from different processes
    min_support : int
        Minimum support threshold
    hash_table_size : int
        Size of the hash table

    Returns
    -------
    frequent_itemsets : list
        List containing the frequent itemsets
    """
    frequent_itemsets = []

    candidate_counts = {}
    hash_table = np.zeros((hash_table_size,), dtype=int)

    for mapped_itemsets, mapped_hash_table in mapped_data:
        for itemset, count in mapped_itemsets.items():
            candidate_counts[itemset] = candidate_counts.get(itemset, 0) + count
            hash_value = hash(itemset) % hash_table_size
            hash_table[hash_value] += count

    # Filter frequent itemsets based on minimum support and hash table counts
    frequent_itemsets = [(itemset, count) for itemset, count in candidate_counts.items() if count >= min_support and hash_table[hash(itemset) % hash_table_size] >= min_support]

    return frequent_itemsets

def PCY(transactions, min_support, hash_table_size, chunk_size):
    """
    PCY algorithm

    Parameters
    ----------
    transactions : list 
        List containing the transactions
    min_support : int
        Minimum support threshold
    hash_table_size : int
        Size of the hash table
    chunk_size : int
        Size of the chunks

    Returns
    -------
    frequent_itemsets : list
        List containing the frequent itemsets

    Notes
    -----
    The PCY algorithm is a two-phase algorithm. In the first phase, frequent single items are counted. 
    In the second phase, candidate pairs are counted using a hash table. 
    The hash table is used to count the candidate pairs and to filter the frequent itemsets based on the hash table counts.
    """
    # First pass - Counting frequent single items
    single_item_counts = {}
    for transaction in transactions:
        for item in transaction:
            single_item_counts[item] = single_item_counts.get(item, 0) + 1

    frequent_single_items = set(
        [item for item, count in single_item_counts.items() if count >= min_support]
    )

    # Divide transactions into chunks
    chunks = [transactions[i:i+chunk_size] for i in range(0, len(transactions), chunk_size)]
    args = [(chunk, frequent_single_items, hash_table_size) for chunk in chunks]

    # Create a pool of worker processes
    print("Number of worker processes: ", cpu_count()//2)
    pool = Pool(cpu_count()//2)

    # Map phase with progress bar
    mapped_data = list(tqdm(pool.imap(mapper, args), total=len(args), desc="Mapping"))

    # Reduce phase with progress bar
    frequent_itemsets = list(tqdm(reducer(mapped_data, min_support, hash_table_size), desc="Reducing"))

    # Close the pool of worker processes
    pool.close()
    pool.join()

    return frequent_itemsets


if __name__ == '__main__':
    transactions = []
    with open('transactions.txt', 'r') as f:
        for line in f:
            transactions.append(line.split())
    
    print("Number of transactions: ", len(transactions))

    min_support = 10000
    hash_table_size = 1000
    chunk_size = 10000

    # Metrics for performance evaluation
    times = []
    sizes = [1000, 10000, 100000, 200000, len(transactions)]
    
    for size in sizes:
        print("Bucket size: ", size)
        start = time.time()
        frequent_itemsets = PCY(transactions[:size], min_support, hash_table_size, chunk_size)
        end = time.time()
        times.append(end-start)
        print("Time taken: ", end-start)
        print()

    # Save the performance metrics to a file (pkl)
    pkl.dump(times, open("pcy_times.pkl", "wb"))



    # Save the frequent itemsets to a file (pkl)
    pkl.dump(frequent_itemsets, open("frequent_itemsets_pcy.pkl", "wb"))

    print("Number of frequent itemsets: ", len(frequent_itemsets))