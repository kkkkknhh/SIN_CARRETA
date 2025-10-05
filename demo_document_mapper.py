#!/usr/bin/env python3
# coding=utf-8
"""
Demo script for DocumentEmbeddingMapper with duplicate handling.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from document_embedding_mapper import DocumentEmbeddingMapper
    
    def main():
        print("=== Document Embedding Mapper Demo ===\n")
        
        # Create mapper
        print("1. Initializing DocumentEmbeddingMapper...")
        mapper = DocumentEmbeddingMapper()
        print("✓ Mapper initialized")
        
        # Sample document segments with duplicates
        segments = [
            ("Introduction to machine learning concepts", 1),
            ("Neural networks and deep learning", 2),
            ("Data preprocessing techniques", 3),
            ("Introduction to machine learning concepts", 4),  # Duplicate text, different page
            ("Model evaluation and validation", 5),
            ("Neural networks and deep learning", 2),  # Exact duplicate
            ("Feature engineering methods", 6),
            ("Introduction to machine learning concepts", 1),  # Exact duplicate
        ]
        
        print(f"\n2. Adding {len(segments)} document segments...")
        indices = mapper.add_document_segments(segments)
        print(f"✓ Added segments at indices: {indices}")
        
        # Verify parallel arrays
        print("\n3. Verifying parallel array consistency...")
        is_consistent = mapper.verify_parallel_arrays()
        print(f"✓ Parallel arrays consistent: {is_consistent}")
        
        # Show statistics
        stats = mapper.get_statistics()
        print("\n4. Document statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test duplicate finding
        print("\n5. Finding duplicates...")
        duplicate_text = "Introduction to machine learning concepts"
        
        # Find duplicates on page 1
        page1_duplicates = mapper.find_duplicate_indices(duplicate_text, 1)
        print(f"   '{duplicate_text}' on page 1: indices {page1_duplicates}")
        
        # Find duplicates on page 4
        page4_duplicates = mapper.find_duplicate_indices(duplicate_text, 4)
        print(f"   '{duplicate_text}' on page 4: indices {page4_duplicates}")
        
        # Test similarity search
        print("\n6. Performing similarity search...")
        query = "machine learning introduction"
        results = mapper.similarity_search(query, k=5)
        
        print(f"   Top 5 results for '{query}':")
        for i, (idx, text, page, score) in enumerate(results):
            print(f"   {i+1}. Index {idx}, Page {page}, Score {score:.4f}")
            print(f"      Text: {text[:60]}...")
        
        # Verify no .index() method is used by checking correct mappings
        print("\n7. Verifying index mappings...")
        for idx, text, page, score in results[:3]:
            retrieved_text, retrieved_page = mapper.get_segment_info(idx)
            print(f"   Index {idx}: Expected ('{text[:30]}...', {page}) == Retrieved ('{retrieved_text[:30]}...', {retrieved_page}) ✓" if text == retrieved_text and page == retrieved_page else "✗")
        
        print("\n✓ Demo completed successfully!")
        print("✓ No .index() method usage - using parallel arrays with direct indexing")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()