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

func minioConcat(a, b interface{}) string {
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

func minioCall(f interface{}, arg interface{}) interface{} {
	// Simple function call handler - for now just return nil
	// In a full implementation, this would handle function values
	return nil
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
