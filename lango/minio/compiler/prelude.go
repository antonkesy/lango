package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Runtime support functions

// Special infinity value
type MinioInfinity struct{}

var minioInfinity = MinioInfinity{}

func minioAdd(a, b interface{}) interface{} {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av + bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av + bv
		}
	}
	return nil
}

func minioSub(a, b interface{}) interface{} {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av - bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av - bv
		}
	}
	return nil
}

func minioMul(a, b interface{}) interface{} {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av * bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av * bv
		}
	}
	return nil
}

func minioDiv(a, b interface{}) interface{} {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			if bv == 0 {
				return minioInfinity
			}
			// Convert to float64 for division to match Minio semantics
			return float64(av) / float64(bv)
		}
	case float64:
		if bv, ok := b.(float64); ok {
			if bv == 0.0 {
				return minioInfinity
			}
			return av / bv
		}
	}
	return nil
}

func minioLessThan(a, b interface{}) bool {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av < bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av < bv
		}
	case string:
		if bv, ok := b.(string); ok {
			return av < bv
		}
	}
	return false
}

func minioLessEqual(a, b interface{}) bool {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av <= bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av <= bv
		}
	case string:
		if bv, ok := b.(string); ok {
			return av <= bv
		}
	}
	return false
}

func minioGreaterThan(a, b interface{}) bool {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av > bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av > bv
		}
	case string:
		if bv, ok := b.(string); ok {
			return av > bv
		}
	}
	return false
}

func minioGreaterEqual(a, b interface{}) bool {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av >= bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av >= bv
		}
	case string:
		if bv, ok := b.(string); ok {
			return av >= bv
		}
	}
	return false
}

func minioEqual(a, b interface{}) bool {
	switch av := a.(type) {
	case int:
		if bv, ok := b.(int); ok {
			return av == bv
		}
	case float64:
		if bv, ok := b.(float64); ok {
			return av == bv
		}
	case string:
		if bv, ok := b.(string); ok {
			return av == bv
		}
	case bool:
		if bv, ok := b.(bool); ok {
			return av == bv
		}
	case []interface{}:
		if bv, ok := b.([]interface{}); ok {
			if len(av) != len(bv) {
				return false
			}
			for i := range av {
				if !minioEqual(av[i], bv[i]) {
					return false
				}
			}
			return true
		}
	}
	return false
}

func minioNotEqual(a, b interface{}) bool {
	return !minioEqual(a, b)
}

func minioConcat(a, b interface{}) interface{} {
	// Check if both operands are lists
	if aList, aOk := a.([]interface{}); aOk {
		if bList, bOk := b.([]interface{}); bOk {
			// List concatenation
			result := make([]interface{}, len(aList)+len(bList))
			copy(result, aList)
			copy(result[len(aList):], bList)
			return result
		}
	}

	// String concatenation (fallback)
	aStr := ""
	bStr := ""
	switch av := a.(type) {
	case string:
		aStr = av
	default:
		aStr = minioShow(av)
	}
	switch bv := b.(type) {
	case string:
		bStr = bv
	default:
		bStr = minioShow(bv)
	}
	return aStr + bStr
}

func minioShow(value interface{}) string {
	switch v := value.(type) {
	case MinioInfinity:
		return "Infinity"
	case bool:
		if v {
			return "True"
		} else {
			return "False"
		}
	case string:
		return `"` + v + `"`
	case []interface{}:
		elements := make([]string, len(v))
		for i, elem := range v {
			elements[i] = minioShow(elem)
		}
		return "[" + strings.Join(elements, ",") + "]"
	case float64:
		// Format float64 to always show at least one decimal place for whole numbers
		if v == float64(int(v)) {
			return strconv.FormatFloat(v, 'f', 1, 64)
		}
		return strconv.FormatFloat(v, 'f', -1, 64)
	case int:
		return strconv.Itoa(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func minioPutStr(s interface{}) interface{} {
	if str, ok := s.(string); ok {
		fmt.Print(str)
	} else {
		fmt.Print(s)
	}
	return nil
}

func minioError(message string) interface{} {
	panic("Runtime error: " + message)
}

// Forward declarations for functions that might be used as first-class values
var minioFunctionRegistry = make(map[interface{}]func(interface{}) interface{})

func init() {
	// This will be populated by generated code
}

func minioCall(f interface{}, arg interface{}) interface{} {
	// Handle function calls by function name or function pointer

	// Try function pointer first
	if fn, ok := f.(func(interface{}) interface{}); ok {
		return fn(arg)
	}

	// If f is a function that takes multiple arguments, we need to handle variadic calls
	if fn, ok := f.(func(...interface{}) interface{}); ok {
		return fn(arg)
	}

	// Try function registry (for first-class function names)
	if fn, ok := minioFunctionRegistry[f]; ok {
		return fn(arg)
	}

	// For specific known functions, dispatch directly
	// This is a workaround until we have proper first-class function support
	switch f {
	// We'll need to add cases for each function that might be used as a first-class value
	// For now, return nil for unhandled cases
	default:
		return nil
	}
}

func minioBool(value interface{}) bool {
	switch v := value.(type) {
	case bool:
		return v
	case int:
		return v != 0
	case float64:
		return v != 0.0
	case string:
		return v != ""
	case []interface{}:
		return len(v) > 0
	case nil:
		return false
	default:
		return true // Non-nil/non-zero values are truthy
	}
}
