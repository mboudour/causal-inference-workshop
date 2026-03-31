# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Still needed for Intel Macs

try:
    from scipy.sparse import csr_matrix
    import sklearn
    from sentence_transformers import SentenceTransformer
    print("✅ System libraries linked successfully.")
except ImportError as e:
    print(f"❌ Still a linking error: {e}")