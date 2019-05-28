/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.search;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermStates;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.ThreadInterruptedException;

/** Implements search over a single IndexReader.
 *
 * <p>Applications usually need only call the inherited
 * {@link #search(Query,int)} method. For
 * performance reasons, if your index is unchanging, you
 * should share a single IndexSearcher instance across
 * multiple searches instead of creating a new one
 * per-search.  If your index has changed and you wish to
 * see the changes reflected in searching, you should
 * use {@link DirectoryReader#openIfChanged(DirectoryReader)}
 * to obtain a new reader and
 * then create a new IndexSearcher from that.  Also, for
 * low-latency turnaround it's best to use a near-real-time
 * reader ({@link DirectoryReader#open(IndexWriter)}).
 * Once you have a new {@link IndexReader}, it's relatively
 * cheap to create a new IndexSearcher from it.
 *
 * <p><b>NOTE</b>: The {@link #search} and {@link #searchAfter} methods are
 * configured to only count top hits accurately up to {@code 1,000} and may
 * return a {@link TotalHits.Relation lower bound} of the hit count if the
 * hit count is greater than or equal to {@code 1,000}. On queries that match
 * lots of documents, counting the number of hits may take much longer than
 * computing the top hits so this trade-off allows to get some minimal
 * information about the hit count without slowing down search too much. The
 * {@link TopDocs#scoreDocs} array is always accurate however. If this behavior
 * doesn't suit your needs, you should create collectors manually with either
 * {@link TopScoreDocCollector#create} or {@link TopFieldCollector#create} and
 * call {@link #search(Query, Collector)}.
 *
 * <a name="thread-safety"></a><p><b>NOTE</b>: <code>{@link
 * IndexSearcher}</code> instances are completely
 * thread safe, meaning multiple threads can call any of its
 * methods, concurrently.  If your application requires
 * external synchronization, you should <b>not</b>
 * synchronize on the <code>IndexSearcher</code> instance;
 * use your own (non-Lucene) objects instead.</p>
 */
public class IndexSearcher {

  private static QueryCache DEFAULT_QUERY_CACHE;
  private static QueryCachingPolicy DEFAULT_CACHING_POLICY = new UsageTrackingQueryCachingPolicy();
  static {
    final int maxCachedQueries = 1000;
    // min of 32MB or 5% of the heap size
    final long maxRamBytesUsed = Math.min(1L << 25, Runtime.getRuntime().maxMemory() / 20);
    DEFAULT_QUERY_CACHE = new LRUQueryCache(maxCachedQueries, maxRamBytesUsed);
  }
  /**
   * By default we count hits accurately up to 1000. This makes sure that we
   * don't spend most time on computing hit counts
   */
  private static final int TOTAL_HITS_THRESHOLD = 1000;

  final IndexReader reader; // package private for testing!
  
  // NOTE: these members might change in incompatible ways
  // in the next release
  protected final IndexReaderContext readerContext;
  protected final List<LeafReaderContext> leafContexts;
  /** used with executor - each slice holds a set of leafs executed within one thread */
  private final LeafSlice[] leafSlices;

  // These are only used for multi-threaded search
  private final ExecutorService executor;

  // the default Similarity
  private static final Similarity defaultSimilarity = new BM25Similarity();

  private QueryCache queryCache = DEFAULT_QUERY_CACHE;
  private QueryCachingPolicy queryCachingPolicy = DEFAULT_CACHING_POLICY;

  /**
   * Expert: returns a default Similarity instance.
   * In general, this method is only called to initialize searchers and writers.
   * User code and query implementations should respect
   * {@link IndexSearcher#getSimilarity()}.
   * @lucene.internal
   */
  public static Similarity getDefaultSimilarity() {
    return defaultSimilarity;
  }

  /**
   * Expert: Get the default {@link QueryCache} or {@code null} if the cache is disabled.
   * @lucene.internal
   */
  public static QueryCache getDefaultQueryCache() {
    return DEFAULT_QUERY_CACHE;
  }

  /**
   * Expert: set the default {@link QueryCache} instance.
   * @lucene.internal
   */
  public static void setDefaultQueryCache(QueryCache defaultQueryCache) {
    DEFAULT_QUERY_CACHE = defaultQueryCache;
  }

  /**
   * Expert: Get the default {@link QueryCachingPolicy}.
   * @lucene.internal
   */
  public static QueryCachingPolicy getDefaultQueryCachingPolicy() {
    return DEFAULT_CACHING_POLICY;
  }

  /**
   * Expert: set the default {@link QueryCachingPolicy} instance.
   * @lucene.internal
   */
  public static void setDefaultQueryCachingPolicy(QueryCachingPolicy defaultQueryCachingPolicy) {
    DEFAULT_CACHING_POLICY = defaultQueryCachingPolicy;
  }

  /** The Similarity implementation used by this searcher. */
  private Similarity similarity = defaultSimilarity;

  /** Creates a searcher searching the provided index. */
  public IndexSearcher(IndexReader r) {
    this(r, null);
  }

  /** Runs searches for each segment separately, using the
   *  provided ExecutorService.  IndexSearcher will not
   *  close/awaitTermination this ExecutorService on
   *  close; you must do so, eventually, on your own.  NOTE:
   *  if you are using {@link NIOFSDirectory}, do not use
   *  the shutdownNow method of ExecutorService as this uses
   *  Thread.interrupt under-the-hood which can silently
   *  close file descriptors (see <a
   *  href="https://issues.apache.org/jira/browse/LUCENE-2239">LUCENE-2239</a>).
   * 
   * @lucene.experimental */
  public IndexSearcher(IndexReader r, ExecutorService executor) {
    this(r.getContext(), executor);
  }

  /**
   * Creates a searcher searching the provided top-level {@link IndexReaderContext}.
   * <p>
   * Given a non-<code>null</code> {@link ExecutorService} this method runs
   * searches for each segment separately, using the provided ExecutorService.
   * IndexSearcher will not close/awaitTermination this ExecutorService on
   * close; you must do so, eventually, on your own. NOTE: if you are using
   * {@link NIOFSDirectory}, do not use the shutdownNow method of
   * ExecutorService as this uses Thread.interrupt under-the-hood which can
   * silently close file descriptors (see <a
   * href="https://issues.apache.org/jira/browse/LUCENE-2239">LUCENE-2239</a>).
   * 
   * @see IndexReaderContext
   * @see IndexReader#getContext()
   * @lucene.experimental
   */
  public IndexSearcher(IndexReaderContext context, ExecutorService executor) {
    assert context.isTopLevel: "IndexSearcher's ReaderContext must be topLevel for reader" + context.reader();
    reader = context.reader();
    this.executor = executor;
    this.readerContext = context;
    leafContexts = context.leaves();
    this.leafSlices = executor == null ? null : slices(leafContexts);
  }

  /**
   * Creates a searcher searching the provided top-level {@link IndexReaderContext}.
   *
   * @see IndexReaderContext
   * @see IndexReader#getContext()
   * @lucene.experimental
   */
  public IndexSearcher(IndexReaderContext context) {
    this(context, null);
  }

  /**
   * Set the {@link QueryCache} to use when scores are not needed.
   * A value of {@code null} indicates that query matches should never be
   * cached. This method should be called <b>before</b> starting using this
   * {@link IndexSearcher}.
   * <p>NOTE: When using a query cache, queries should not be modified after
   * they have been passed to IndexSearcher.
   * @see QueryCache
   * @lucene.experimental
   */
  public void setQueryCache(QueryCache queryCache) {
    this.queryCache = queryCache;
  }

  /**
   * Return the query cache of this {@link IndexSearcher}. This will be either
   * the {@link #getDefaultQueryCache() default query cache} or the query cache
   * that was last set through {@link #setQueryCache(QueryCache)}. A return
   * value of {@code null} indicates that caching is disabled.
   * @lucene.experimental
   */
  public QueryCache getQueryCache() {
    return queryCache;
  }

  /**
   * Set the {@link QueryCachingPolicy} to use for query caching.
   * This method should be called <b>before</b> starting using this
   * {@link IndexSearcher}.
   * @see QueryCachingPolicy
   * @lucene.experimental
   */
  public void setQueryCachingPolicy(QueryCachingPolicy queryCachingPolicy) {
    this.queryCachingPolicy = Objects.requireNonNull(queryCachingPolicy);
  }

  /**
   * Return the query cache of this {@link IndexSearcher}. This will be either
   * the {@link #getDefaultQueryCachingPolicy() default policy} or the policy
   * that was last set through {@link #setQueryCachingPolicy(QueryCachingPolicy)}.
   * @lucene.experimental
   */
  public QueryCachingPolicy getQueryCachingPolicy() {
    return queryCachingPolicy;
  }

  /**
   * Expert: Creates an array of leaf slices each holding a subset of the given leaves.
   * Each {@link LeafSlice} is executed in a single thread. By default there
   * will be one {@link LeafSlice} per leaf ({@link org.apache.lucene.index.LeafReaderContext}).
   */
  protected LeafSlice[] slices(List<LeafReaderContext> leaves) {
    LeafSlice[] slices = new LeafSlice[leaves.size()];
    for (int i = 0; i < slices.length; i++) {
      slices[i] = new LeafSlice(leaves.get(i));
    }
    return slices;
  }
  
  /** Return the {@link IndexReader} this searches. */
  public IndexReader getIndexReader() {
    return reader;
  }

  /** 
   * Sugar for <code>.getIndexReader().document(docID)</code> 
   * @see IndexReader#document(int) 
   */
  public Document doc(int docID) throws IOException {
    return reader.document(docID);
  }

  /** 
   * Sugar for <code>.getIndexReader().document(docID, fieldVisitor)</code>
   * @see IndexReader#document(int, StoredFieldVisitor) 
   */
  public void doc(int docID, StoredFieldVisitor fieldVisitor) throws IOException {
    reader.document(docID, fieldVisitor);
  }

  /** 
   * Sugar for <code>.getIndexReader().document(docID, fieldsToLoad)</code>
   * @see IndexReader#document(int, Set) 
   */
  public Document doc(int docID, Set<String> fieldsToLoad) throws IOException {
    return reader.document(docID, fieldsToLoad);
  }

  /** Expert: Set the Similarity implementation used by this IndexSearcher.
   *
   */
  public void setSimilarity(Similarity similarity) {
    this.similarity = similarity;
  }

  /** Expert: Get the {@link Similarity} to use to compute scores. This returns the
   *  {@link Similarity} that has been set through {@link #setSimilarity(Similarity)}
   *  or the default {@link Similarity} if none has been set explicitly. */
  public Similarity getSimilarity() {
    return similarity;
  }

  /**
   * Count how many documents match the given query.
   */
  public int count(Query query) throws IOException {
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	System.out.println(ste);
		//}
		//System.out.println("[arcj] IndexSeacher::count");

    query = rewrite(query);
    while (true) {
      // remove wrappers that don't matter for counts
      if (query instanceof ConstantScoreQuery) {
        query = ((ConstantScoreQuery) query).getQuery();
      } else {
        break;
      }
    }

    // some counts can be computed in constant time
    if (query instanceof MatchAllDocsQuery) {
      return reader.numDocs();
    } else if (query instanceof TermQuery && reader.hasDeletions() == false) {
      Term term = ((TermQuery) query).getTerm();
      int count = 0;
      for (LeafReaderContext leaf : reader.leaves()) {
        count += leaf.reader().docFreq(term);
      }
      return count;
    }

    // general case: create a collecor and count matches
    final CollectorManager<TotalHitCountCollector, Integer> collectorManager = new CollectorManager<TotalHitCountCollector, Integer>() {

      @Override
      public TotalHitCountCollector newCollector() throws IOException {
        return new TotalHitCountCollector();
      }

      @Override
      public Integer reduce(Collection<TotalHitCountCollector> collectors) throws IOException {
        int total = 0;
        for (TotalHitCountCollector collector : collectors) {
          total += collector.getTotalHits();
        }
        return total;
      }

    };
    return search(query, collectorManager);
  }

  /** Returns the leaf slices used for concurrent searching, or null if no {@code ExecutorService} was
   *  passed to the constructor.
   *
   * @lucene.experimental */
  public LeafSlice[] getSlices() {
      return leafSlices;
  }
  
  /** Finds the top <code>n</code>
   * hits for <code>query</code> where all results are after a previous 
   * result (<code>after</code>).
   * <p>
   * By passing the bottom result from a previous page as <code>after</code>,
   * this method can be used for efficient 'deep-paging' across potentially
   * large result sets.
   *
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public TopDocs searchAfter(ScoreDoc after, Query query, int numHits) throws IOException {
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	System.out.println(ste);
		//}
		//System.out.println("[arcj] searchafter1");
		long start1 = System.nanoTime();
		//long start2 = System.nanoTime();
    final int limit = Math.max(1, reader.maxDoc());

    if (after != null && after.doc >= limit) {
      throw new IllegalArgumentException("after.doc exceeds the number of documents in the reader: after.doc="
          + after.doc + " limit=" + limit);
    }

    final int cappedNumHits = Math.min(numHits, limit);
		//System.out.println("[arcj] searchafter1 " + numHits + " " + limit );
		//long end2 = System.nanoTime();
		//long timeElapsed2 = end2 - start2;
		//System.out.println("[arcj] searchafter1 others1 " + timeElapsed2);
		//System.out.println("[arcj] searchafter1 others2 ");
		//long start3 = System.nanoTime();
    final CollectorManager<TopScoreDocCollector, TopDocs> manager = new CollectorManager<TopScoreDocCollector, TopDocs>() {

      @Override
      public TopScoreDocCollector newCollector() throws IOException {
				// arcj: create
				//long start5 = System.nanoTime();
				TopScoreDocCollector tmp1 = TopScoreDocCollector.create(cappedNumHits, after, TOTAL_HITS_THRESHOLD);
				System.out.println("[arcj] searchafter1 create " + cappedNumHits + " / " + after + " / " + TOTAL_HITS_THRESHOLD + " / " + tmp1);
				//long end5 = System.nanoTime();
				//long timeElapsed5 = end5 - start5;
				//System.out.println("[arcj] searchafter1 create " + timeElapsed5);

				return tmp1;
        //return TopScoreDocCollector.create(cappedNumHits, after, TOTAL_HITS_THRESHOLD);
      }

			// arcj: reduce
      @Override
      public TopDocs reduce(Collection<TopScoreDocCollector> collectors) throws IOException {
        final TopDocs[] topDocs = new TopDocs[collectors.size()];
        int i = 0;
				int z = 0;

				// arcj: points to the top N search results which matches the search criteria
				//long start6 = System.nanoTime();
        for (TopScoreDocCollector collector : collectors) {
          topDocs[i++] = collector.topDocs();
					//System.out.println("[arcj] searchafter1 topdocs " + topDocs[z++]);
        }
				//long end6 = System.nanoTime();
				//long timeElapsed6 = end6 - start6;
				//System.out.println("[arcj] searchafter1 topdocs " + timeElapsed6);

				// arcj: merge
				//System.out.println("[arcj] searchafter1 merge ");
				//long start7 = System.nanoTime();
				TopDocs tmp2 = TopDocs.merge(0, cappedNumHits, topDocs, true);
				//long end7 = System.nanoTime();
				//long timeElapsed7 = end7 - start7;
				//System.out.println("[arcj] searchafter1 merge " + timeElapsed7);

				return tmp2;
        //return TopDocs.merge(0, cappedNumHits, topDocs, true);
      }
    };
		//long end3 = System.nanoTime();
		//long timeElapsed3 = end3 - start3;
		//System.out.println("[arcj] searchafter1 others2 " + timeElapsed3);

		// arcj: search
		System.out.println("[arcj] searchafter1 search ");
		long start4 = System.nanoTime();
		TopDocs tmp3 = search(query, manager); // search5
		long end4 = System.nanoTime();
		long timeElapsed4 = end4 - start4;
		System.out.println("[arcj] searchafter1 search " + timeElapsed4);

		// arcj: all 1
		long end1 = System.nanoTime();
		long timeElapsed1 = end1 - start1;
		System.out.println("[arcj] searchafter1 all " + timeElapsed1);
		return tmp3;
    //return search(query, manager);
  }

  /** Finds the top <code>n</code>
   * hits for <code>query</code>.
   *
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public TopDocs search(Query query, int n)
    throws IOException {
		System.out.println("");
		System.out.println("[arcj] search1 " + query + " / " + n);

		long start = System.nanoTime();
		TopDocs tmp = searchAfter(null, query, n);
		long end = System.nanoTime();
		long timeElapsed = end - start;
		System.out.println("[arcj] search1 end " + timeElapsed);
	
		return tmp;
		//return searchAfter(null, query, n);
  }

  /** Lower-level search API.
   *
   * <p>{@link LeafCollector#collect(int)} is called for every matching document.
   *
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public void search(Query query, Collector results)
    throws IOException {
		//System.out.println("[arcj] search2 " + query + " / " + leafContexts.size());

		long start1 = System.nanoTime();
    query = rewrite(query);
		long end1 = System.nanoTime();
		long timeElapsed1 = end1 - start1;
		System.out.println("[arcj] search2 rewrite " + timeElapsed1 + " / " + query);

		//System.out.println("[arcj] search2 search before " + leafContexts + " / " + results.scoreMode() + " / " + results);
		
		long start2 = System.nanoTime();
		Weight tmp = createWeight(query, results.scoreMode(), 1);
		long end2 = System.nanoTime();
		long timeElapsed2 = end2 - start2;
		System.out.println("[arcj] search2 createWeight " + timeElapsed2);


		long start3 = System.nanoTime();
    //search(leafContexts, createWeight(query, results.scoreMode(), 1), results); // search6
    search(leafContexts, tmp, results); // search6
		long end3 = System.nanoTime();
		long timeElapsed3 = end3 - start3;
		System.out.println("[arcj] search2 search " + timeElapsed3);
  }

  /** Search implementation with arbitrary sorting, plus
   * control over whether hit scores and max score
   * should be computed.  Finds
   * the top <code>n</code> hits for <code>query</code>, and sorting
   * the hits by the criteria in <code>sort</code>.
   * If <code>doDocScores</code> is <code>true</code>
   * then the score of each hit will be computed and
   * returned.  If <code>doMaxScore</code> is
   * <code>true</code> then the maximum score over all
   * collected hits will be computed.
   * 
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public TopFieldDocs search(Query query, int n,
      Sort sort, boolean doDocScores) throws IOException {
		System.out.println("[arcj] search3");
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	    System.out.println(ste);
		//}

    return searchAfter(null, query, n, sort, doDocScores);
  }

  /**
   * Search implementation with arbitrary sorting.
   * @param query The query to search for
   * @param n Return only the top n results
   * @param sort The {@link org.apache.lucene.search.Sort} object
   * @return The top docs, sorted according to the supplied {@link org.apache.lucene.search.Sort} instance
   * @throws IOException if there is a low-level I/O error
   */
  public TopFieldDocs search(Query query, int n, Sort sort) throws IOException {
		System.out.println("[arcj] search4");
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	    System.out.println(ste);
		//}
		long start = System.nanoTime();
    TopFieldDocs tmp = searchAfter(null, query, n, sort, false);
		long end = System.nanoTime();
		long timeElapsed = end - start;
		System.out.println("[arcj] searchtime4 " + timeElapsed);

		return tmp;
    //return searchAfter(null, query, n, sort, false);
  }

  /** Finds the top <code>n</code>
   * hits for <code>query</code> where all results are after a previous
   * result (<code>after</code>).
   * <p>
   * By passing the bottom result from a previous page as <code>after</code>,
   * this method can be used for efficient 'deep-paging' across potentially
   * large result sets.
   *
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public TopDocs searchAfter(ScoreDoc after, Query query, int n, Sort sort) throws IOException {
		System.out.println("[arcj] searchAfter2");
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	    System.out.println(ste);
		//}

    return searchAfter(after, query, n, sort, false);
  }

  /** Finds the top <code>n</code>
   * hits for <code>query</code> where all results are after a previous
   * result (<code>after</code>), allowing control over
   * whether hit scores and max score should be computed.
   * <p>
   * By passing the bottom result from a previous page as <code>after</code>,
   * this method can be used for efficient 'deep-paging' across potentially
   * large result sets.  If <code>doDocScores</code> is <code>true</code>
   * then the score of each hit will be computed and
   * returned.  If <code>doMaxScore</code> is
   * <code>true</code> then the maximum score over all
   * collected hits will be computed.
   *
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public TopFieldDocs searchAfter(ScoreDoc after, Query query, int numHits, Sort sort,
      boolean doDocScores) throws IOException {
		System.out.println("[arcj] searchAfter3");
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	    System.out.println(ste);
		//}

    if (after != null && !(after instanceof FieldDoc)) {
      // TODO: if we fix type safety of TopFieldDocs we can
      // remove this
      throw new IllegalArgumentException("after must be a FieldDoc; got " + after);
    }
    return searchAfter((FieldDoc) after, query, numHits, sort, doDocScores);
  }

  private TopFieldDocs searchAfter(FieldDoc after, Query query, int numHits, Sort sort,
      boolean doDocScores) throws IOException {
		System.out.println("[arcj] searchAfter4");
		//for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
		//	    System.out.println(ste);
		//}

    final int limit = Math.max(1, reader.maxDoc());
    if (after != null && after.doc >= limit) {
      throw new IllegalArgumentException("after.doc exceeds the number of documents in the reader: after.doc="
          + after.doc + " limit=" + limit);
    }
    final int cappedNumHits = Math.min(numHits, limit);
    final Sort rewrittenSort = sort.rewrite(this);

    final CollectorManager<TopFieldCollector, TopFieldDocs> manager = new CollectorManager<TopFieldCollector, TopFieldDocs>() {

      @Override
      public TopFieldCollector newCollector() throws IOException {
        // TODO: don't pay the price for accurate hit counts by default
        return TopFieldCollector.create(rewrittenSort, cappedNumHits, after, TOTAL_HITS_THRESHOLD);
      }

      @Override
      public TopFieldDocs reduce(Collection<TopFieldCollector> collectors) throws IOException {
        final TopFieldDocs[] topDocs = new TopFieldDocs[collectors.size()];
        int i = 0;
        for (TopFieldCollector collector : collectors) {
          topDocs[i++] = collector.topDocs();
        }
        return TopDocs.merge(rewrittenSort, 0, cappedNumHits, topDocs, true);
      }

    };

    TopFieldDocs topDocs = search(query, manager);
    if (doDocScores) {
      TopFieldCollector.populateScores(topDocs.scoreDocs, this, query);
    }
    return topDocs;
  }

 /**
  * Lower-level search API.
  * Search all leaves using the given {@link CollectorManager}. In contrast
  * to {@link #search(Query, Collector)}, this method will use the searcher's
  * {@link ExecutorService} in order to parallelize execution of the collection
  * on the configured {@link #leafSlices}.
  * @see CollectorManager
  * @lucene.experimental
  */
  public <C extends Collector, T> T search(Query query, CollectorManager<C, T> collectorManager) throws IOException {
		//System.out.println("[arcj] search5 " + query + " / " + collectorManager);
    if (executor == null) {
			long start = System.nanoTime();
      final C collector = collectorManager.newCollector();
			long end = System.nanoTime();
			long timeElapsed = end - start;
			System.out.println("[arcj] search5 create " + timeElapsed + " / " + collector);

			// arcj
			long start1 = System.nanoTime();
      search(query, collector);	// search2
			long end1 = System.nanoTime();
			long timeElapsed1 = end1 - start1;
			System.out.println("[arcj] search5 search " + timeElapsed1);

			// arcj
			long start2 = System.nanoTime();
			T tmp1 = collectorManager.reduce(Collections.singletonList(collector));
			long end2 = System.nanoTime();
			long timeElapsed2 = end2 - start2;
			System.out.println("[arcj] search5 reduce1 " + timeElapsed2);

			return tmp1;
      //return collectorManager.reduce(Collections.singletonList(collector));
    } else {
      final List<C> collectors = new ArrayList<>(leafSlices.length);
      ScoreMode scoreMode = null;
      for (int i = 0; i < leafSlices.length; ++i) {
        final C collector = collectorManager.newCollector();
        collectors.add(collector);
        if (scoreMode == null) {
          scoreMode = collector.scoreMode();
        } else if (scoreMode != collector.scoreMode()) {
          throw new IllegalStateException("CollectorManager does not always produce collectors with the same score mode");
        }
      }
      if (scoreMode == null) {
        // no segments
        scoreMode = ScoreMode.COMPLETE;
      }
			// arcj
			long start3 = System.nanoTime();
      query = rewrite(query);
			long end3 = System.nanoTime();
			long timeElapsed3 = end3 - start3;
			System.out.println("[arcj] search5 rewrite " + timeElapsed3);

      final Weight weight = createWeight(query, scoreMode, 1);
      final List<Future<C>> topDocsFutures = new ArrayList<>(leafSlices.length);

			// arcj
			long start4 = System.nanoTime();
      for (int i = 0; i < leafSlices.length; ++i) {
        final LeafReaderContext[] leaves = leafSlices[i].leaves;
        final C collector = collectors.get(i);
        topDocsFutures.add(executor.submit(new Callable<C>() {
          @Override
          public C call() throws Exception {
            search(Arrays.asList(leaves), weight, collector);
            return collector;
          }
        }));
      }
			long end4 = System.nanoTime();
			long timeElapsed4 = end4 - start4;
			System.out.println("[arcj] search5 leaves " + timeElapsed4);

      final List<C> collectedCollectors = new ArrayList<>();
			// arcj
			long start5 = System.nanoTime();
      for (Future<C> future : topDocsFutures) {
        try {
          collectedCollectors.add(future.get());
        } catch (InterruptedException e) {
          throw new ThreadInterruptedException(e);
        } catch (ExecutionException e) {
          throw new RuntimeException(e);
        }
      }
			long end5 = System.nanoTime();
			long timeElapsed5 = end5 - start5;
			System.out.println("[arcj] search5 topdocsfutures " + timeElapsed5);

			long start6 = System.nanoTime();
			T tmp2 = collectorManager.reduce(collectors);
			long end6 = System.nanoTime();
			long timeElapsed6 = end6 - start6;
			System.out.println("[arcj] search5 reduce2 " + timeElapsed6);
			// arcj
			return tmp2;
      //return collectorManager.reduce(collectors);
    }
  }

  /**
   * Lower-level search API.
   * 
   * <p>
   * {@link LeafCollector#collect(int)} is called for every document. <br>
   * 
   * <p>
   * NOTE: this method executes the searches on all given leaves exclusively.
   * To search across all the searchers leaves use {@link #leafContexts}.
   * 
   * @param leaves 
   *          the searchers leaves to execute the searches on
   * @param weight
   *          to match documents
   * @param collector
   *          to receive hits
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  protected void search(List<LeafReaderContext> leaves, Weight weight, Collector collector)
      throws IOException {
		long startall = System.nanoTime();
		long start1, start2, start3;
		long end1, end2, end3;
		long timeElapsed1 = 0;
		long timeElapsed2 = 0;
		long timeElapsed3 = 0;

		//System.out.println("[arcj] search6-1 " + leaves + " / " + weight + " / " + collector);
    // TODO: should we make this
    // threaded...?  the Collector could be sync'd?
    // always use single thread:
    for (LeafReaderContext ctx : leaves) { // search each subreader
			//System.out.println("[arcj] search6-2-1 " + ctx);

			start1 = System.nanoTime();
      final LeafCollector leafCollector;
      try {
        leafCollector = collector.getLeafCollector(ctx);
				//System.out.println("[arcj] search6-2-2 " + leafCollector);
      } catch (CollectionTerminatedException e) {
        // there is no doc of interest in this reader context
        // continue with the following leaf
        continue;
      }
			end1 = System.nanoTime();
			timeElapsed1 += end1 - start1;

			start2 = System.nanoTime();
      BulkScorer scorer = weight.bulkScorer(ctx);
			end2 = System.nanoTime();
			timeElapsed2 += end2 - start2;

			start3 = System.nanoTime();
			//System.out.println("[arcj] search6-3 ctx: " + ctx + " / " + scorer + " / " + leafCollector);
      if (scorer != null) {
        try {
					//System.out.println("[arcj] search6-4 " + ctx.reader().getLiveDocs());
          scorer.score(leafCollector, ctx.reader().getLiveDocs());
        } catch (CollectionTerminatedException e) {
          // collection was terminated prematurely
          // continue with the following leaf
        }
      }
			end3 = System.nanoTime();
			timeElapsed3 += end3 - start3;

    }
		long endall = System.nanoTime();
		long timeElapsedall = endall - startall;
		System.out.println("[arcj] search6-1  " + timeElapsed1);
		System.out.println("[arcj] search6-2  " + timeElapsed2);
		System.out.println("[arcj] search6-3  " + timeElapsed3);
		System.out.println("[arcj] search6 total " + timeElapsedall);
  }

  /** Expert: called to re-write queries into primitive queries.
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  public Query rewrite(Query original) throws IOException {
    Query query = original;
    for (Query rewrittenQuery = query.rewrite(reader); rewrittenQuery != query;
         rewrittenQuery = query.rewrite(reader)) {
      query = rewrittenQuery;
    }
    return query;
  }

  /** Returns an Explanation that describes how <code>doc</code> scored against
   * <code>query</code>.
   *
   * <p>This is intended to be used in developing Similarity implementations,
   * and, for good performance, should not be displayed with every hit.
   * Computing an explanation is as expensive as executing the query over the
   * entire index.
   */
  public Explanation explain(Query query, int doc) throws IOException {
    query = rewrite(query);
    return explain(createWeight(query, ScoreMode.COMPLETE, 1), doc);
  }

  /** Expert: low-level implementation method
   * Returns an Explanation that describes how <code>doc</code> scored against
   * <code>weight</code>.
   *
   * <p>This is intended to be used in developing Similarity implementations,
   * and, for good performance, should not be displayed with every hit.
   * Computing an explanation is as expensive as executing the query over the
   * entire index.
   * <p>Applications should call {@link IndexSearcher#explain(Query, int)}.
   * @throws BooleanQuery.TooManyClauses If a query would exceed 
   *         {@link BooleanQuery#getMaxClauseCount()} clauses.
   */
  protected Explanation explain(Weight weight, int doc) throws IOException {
    int n = ReaderUtil.subIndex(doc, leafContexts);
    final LeafReaderContext ctx = leafContexts.get(n);
    int deBasedDoc = doc - ctx.docBase;
    final Bits liveDocs = ctx.reader().getLiveDocs();
    if (liveDocs != null && liveDocs.get(deBasedDoc) == false) {
      return Explanation.noMatch("Document " + doc + " is deleted");
    }
    return weight.explain(ctx, deBasedDoc);
  }

  /**
   * Creates a {@link Weight} for the given query, potentially adding caching
   * if possible and configured.
   * @lucene.experimental
   */
  public Weight createWeight(Query query, ScoreMode scoreMode, float boost) throws IOException {
    final QueryCache queryCache = this.queryCache;
    Weight weight = query.createWeight(this, scoreMode, boost);
		//System.out.println("[arcj] IndexSeacher::createWeight " + weight);
		
    if (scoreMode.needsScores() == false && queryCache != null) {
      weight = queryCache.doCache(weight, queryCachingPolicy);
    }
    return weight;
  }

  /**
   * Returns this searchers the top-level {@link IndexReaderContext}.
   * @see IndexReader#getContext()
   */
  /* sugar for #getReader().getTopReaderContext() */
  public IndexReaderContext getTopReaderContext() {
    return readerContext;
  }

  /**
   * A class holding a subset of the {@link IndexSearcher}s leaf contexts to be
   * executed within a single thread.
   * 
   * @lucene.experimental
   */
  public static class LeafSlice {

    /** The leaves that make up this slice.
     *
     *  @lucene.experimental */
    public final LeafReaderContext[] leaves;
    
    public LeafSlice(LeafReaderContext... leaves) {
      this.leaves = leaves;
    }
  }

  @Override
  public String toString() {
    return "IndexSearcher(" + reader + "; executor=" + executor + ")";
  }
  
  /**
   * Returns {@link TermStatistics} for a term, or {@code null} if
   * the term does not exist.
   * 
   * This can be overridden for example, to return a term's statistics
   * across a distributed collection.
   * @lucene.experimental
   */
  public TermStatistics termStatistics(Term term, TermStates context) throws IOException {
    if (context.docFreq() == 0) {
      return null;
    } else {
      return new TermStatistics(term.bytes(), context.docFreq(), context.totalTermFreq());
    }
  }
  
  /**
   * Returns {@link CollectionStatistics} for a field, or {@code null} if
   * the field does not exist (has no indexed terms)
   * 
   * This can be overridden for example, to return a field's statistics
   * across a distributed collection.
   * @lucene.experimental
   */
  public CollectionStatistics collectionStatistics(String field) throws IOException {
    assert field != null;
    long docCount = 0;
    long sumTotalTermFreq = 0;
    long sumDocFreq = 0;
    for (LeafReaderContext leaf : reader.leaves()) {
      final Terms terms = leaf.reader().terms(field);
      if (terms == null) {
        continue;
      }
      docCount += terms.getDocCount();
      sumTotalTermFreq += terms.getSumTotalTermFreq();
      sumDocFreq += terms.getSumDocFreq();
    }
    if (docCount == 0) {
      return null;
    }
    return new CollectionStatistics(field, reader.maxDoc(), docCount, sumTotalTermFreq, sumDocFreq);
  }
}
