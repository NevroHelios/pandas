#!/usr/bin/env python3
"""
Working test script that reproduces the exact failing test cases
"""

import pandas as pd
import pyarrow as pa
from pandas.core.arrays import ArrowExtensionArray

def test_timestamp():
    print("=== Testing Timestamp Case ===")
    
    # Create timestamp
    timestamps = pd.to_datetime(['2020-01-01 01:01:01.000001']).tz_localize('US/Eastern')
    
    # Create with nanosecond precision like in the failing test
    arrow_dtype = pd.ArrowDtype(pa.timestamp('ns', tz='US/Eastern'))
    data_missing = ArrowExtensionArray._from_sequence([pd.NA, timestamps[0]], dtype=arrow_dtype)
    
    print("Original array:")
    print(f"  dtype: {data_missing.dtype}")
    print(f"  pyarrow_dtype: {data_missing.dtype.pyarrow_dtype}")
    print(f"  unit: {data_missing.dtype.pyarrow_dtype.unit}")
    print(f"  timezone: {data_missing.dtype.pyarrow_dtype.tz}")
    print(f"  values: {data_missing}")
    print()
    
    # Test the map operation that's failing
    print("Testing map operation:")
    result = data_missing.map(lambda x: x, na_action='ignore')
    
    print("Result array:")
    print(f"  dtype: {result.dtype}")
    print(f"  pyarrow_dtype: {result.dtype.pyarrow_dtype}")
    print(f"  unit: {result.dtype.pyarrow_dtype.unit}")
    print(f"  timezone: {result.dtype.pyarrow_dtype.tz}")
    print(f"  values: {result}")
    print()
    
    # Check if they're equal (this is what the test is checking)
    dtypes_equal = data_missing.dtype == result.dtype
    print(f"Timestamp dtypes equal: {dtypes_equal}")
    
    if not dtypes_equal:
        print("❌ TIMESTAMP TEST WOULD FAIL!")
        print(f"Expected: {data_missing.dtype}")
        print(f"Got:      {result.dtype}")
    else:
        print("✅ Timestamp test would pass!")
    
    return dtypes_equal

def test_integer():
    print("\n=== Testing Integer Case ===")
    
    # Create integer array like in the failing test
    int_dtype = pd.ArrowDtype(pa.int64())
    data_missing = ArrowExtensionArray._from_sequence([pd.NA, 1], dtype=int_dtype)
    
    print("Original array:")
    print(f"  dtype: {data_missing.dtype}")
    print(f"  pyarrow_dtype: {data_missing.dtype.pyarrow_dtype}")
    print(f"  values: {data_missing}")
    print(f"  _pa_array.type: {data_missing._pa_array.type}")
    print()
    
    # Test the map operation
    print("Testing map operation:")
    result = data_missing.map(lambda x: x, na_action='ignore')
    
    print("Result array:")
    print(f"  dtype: {result.dtype}")
    print(f"  pyarrow_dtype: {result.dtype.pyarrow_dtype}")
    print(f"  values: {result}")
    print()
    
    # Check if they're equal
    dtypes_equal = data_missing.dtype == result.dtype
    print(f"Integer dtypes equal: {dtypes_equal}")
    
    if not dtypes_equal:
        print("❌ INTEGER TEST WOULD FAIL!")
        print(f"Expected: {data_missing.dtype}")
        print(f"Got:      {result.dtype}")
    else:
        print("✅ Integer test would pass!")
    
    return dtypes_equal

def test_cast_pointwise_directly():
    print("\n=== Testing _cast_pointwise_result directly ===")
    
    # Test with timestamp
    print("Testing timestamp cast:")
    timestamps = pd.to_datetime(['2020-01-01 01:01:01.000001']).tz_localize('US/Eastern')
    arrow_dtype_ns = pd.ArrowDtype(pa.timestamp('ns', tz='US/Eastern'))
    data_ns = ArrowExtensionArray._from_sequence([pd.NA, timestamps[0]], dtype=arrow_dtype_ns)
    
    arrow_dtype_us = pd.ArrowDtype(pa.timestamp('us', tz='US/Eastern'))
    data_us = ArrowExtensionArray._from_sequence([pd.NA, timestamps[0]], dtype=arrow_dtype_us)
    
    print(f"Original (ns): {data_ns.dtype}")
    print(f"Wrong (us):    {data_us.dtype}")
    
    try:
        fixed_result = data_ns._cast_pointwise_result(data_us)
        print(f"Fixed result:  {fixed_result.dtype}")
        print(f"Timestamp fix works: {data_ns.dtype == fixed_result.dtype}")
    except Exception as e:
        print(f"Timestamp cast error: {e}")
    
    # Test with integer
    print("\nTesting integer cast:")
    int_dtype = pd.ArrowDtype(pa.int64())
    data_int = ArrowExtensionArray._from_sequence([pd.NA, 1], dtype=int_dtype)
    
    double_dtype = pd.ArrowDtype(pa.float64())
    data_double = ArrowExtensionArray._from_sequence([pd.NA, 1.0], dtype=double_dtype)
    
    print(f"Original (int64): {data_int.dtype}")
    print(f"Wrong (double):   {data_double.dtype}")
    
    try:
        fixed_result = data_int._cast_pointwise_result(data_double)
        print(f"Fixed result:     {fixed_result.dtype}")
        print(f"Integer fix works: {data_int.dtype == fixed_result.dtype}")
    except Exception as e:
        print(f"Integer cast error: {e}")

def debug_pa_array_creation():
    print("\n=== Debugging pa.array() behavior ===")
    
    # Test what happens when we create pa.array from integer values
    values_int = [None, 1]
    values_float = [None, 1.0]
    
    print("Testing pa.array with integer values:")
    arr_int = pa.array(values_int, from_pandas=True)
    print(f"  Input: {values_int}")
    print(f"  Result type: {arr_int.type}")
    
    print("Testing pa.array with float values:")  
    arr_float = pa.array(values_float, from_pandas=True)
    print(f"  Input: {values_float}")
    print(f"  Result type: {arr_float.type}")
    
    # Test mixed values (this might be the issue)
    mixed_values = [pd.NA, 1]
    print("Testing pa.array with mixed NA/int values:")
    arr_mixed = pa.array(mixed_values, from_pandas=True)
    print(f"  Input: {mixed_values}")
    print(f"  Result type: {arr_mixed.type}")

if __name__ == "__main__":
    print("Testing Arrow dtype preservation issues...")
    print("=" * 60)
    
    # Run all tests
    ts_pass = test_timestamp()
    int_pass = test_integer()
    test_cast_pointwise_directly()
    debug_pa_array_creation()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Timestamp test: {'✅ PASS' if ts_pass else '❌ FAIL'}")
    print(f"Integer test:   {'✅ PASS' if int_pass else '❌ FAIL'}")