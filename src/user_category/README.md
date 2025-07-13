## Notes
1. Handle user category cluster shifting on new transaction data arrival? 
   - Static cluster assignment. 
   Assign each wallet to a cluster once and store it. 
   Never change its cluster — even if its behavior changes. 
     - ✅ Good if you care about “historical” labeling — e.g., “What cluster was this wallet when I first profiled it?” 
     - ❌ Bad if you care about up-to-date behavior, because the label gets stale.

   - Dynamic cluster assignment. 
   Recompute the wallet’s features whenever it does new transactions. 
   Use your saved cluster centroids to predict its new cluster.
     - ✅ Good if you want real-time, accurate behavior grouping.
     - ❌ More work — you need pipelines to recalculate features and reassign clusters.

   In practice, dynamic is better for most queue prioritization or anomaly detection tasks — because you want the cluster to reflect the wallet’s latest behavior.
2. Automatic cluster interpretation?
3. I am using asset_id as one of the identifier. 
   That is because I don't have asset price data to convert the amount into common currency.
   Is it correct?
4. Filtering features using proxy from clustering and re-cluster, repeat until it produces similar results.
5. Get domain experts opinion on the results.
