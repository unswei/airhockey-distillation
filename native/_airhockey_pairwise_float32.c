#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>
#include <string.h>

#define MAX_INPUT_WIDTH 96

static int require_float32_buffer(
    PyObject *object,
    Py_buffer *view,
    int dimensions,
    int writable,
    const char *name
) {
    int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES;
    if (writable) {
        flags |= PyBUF_WRITABLE;
    }
    if (PyObject_GetBuffer(object, view, flags) < 0) {
        return -1;
    }
    if (view->ndim != dimensions ||
        view->itemsize != (Py_ssize_t)sizeof(float) ||
        view->format == NULL || strcmp(view->format, "f") != 0) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be a %d-D float32 buffer",
            name,
            dimensions
        );
        PyBuffer_Release(view);
        return -1;
    }
    for (int axis = 0; axis < dimensions; ++axis) {
        Py_ssize_t expected_stride = (Py_ssize_t)sizeof(float);
        for (int inner = dimensions - 1; inner > axis; --inner) {
            expected_stride *= view->shape[inner];
        }
        if (view->strides[axis] != expected_stride) {
            PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
            PyBuffer_Release(view);
            return -1;
        }
    }
    return 0;
}

static PyObject *linear_into(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *value_object;
    PyObject *weight_object;
    PyObject *bias_object;
    PyObject *output_object;
    if (!PyArg_ParseTuple(
            args,
            "OOOO:linear_into",
            &value_object,
            &weight_object,
            &bias_object,
            &output_object
        )) {
        return NULL;
    }

    Py_buffer value = {0};
    Py_buffer weight = {0};
    Py_buffer bias = {0};
    Py_buffer output = {0};
    int have_bias = bias_object != Py_None;
    if (require_float32_buffer(value_object, &value, 1, 0, "value") < 0) {
        return NULL;
    }
    if (require_float32_buffer(weight_object, &weight, 2, 0, "weight") < 0) {
        PyBuffer_Release(&value);
        return NULL;
    }
    if (have_bias &&
        require_float32_buffer(bias_object, &bias, 1, 0, "bias") < 0) {
        PyBuffer_Release(&weight);
        PyBuffer_Release(&value);
        return NULL;
    }
    if (require_float32_buffer(
            output_object,
            &output,
            1,
            1,
            "output"
        ) < 0) {
        if (have_bias) {
            PyBuffer_Release(&bias);
        }
        PyBuffer_Release(&weight);
        PyBuffer_Release(&value);
        return NULL;
    }

    Py_ssize_t output_width = weight.shape[0];
    Py_ssize_t input_width = weight.shape[1];
    if (input_width <= 0 || input_width > MAX_INPUT_WIDTH ||
        value.shape[0] != input_width || output.shape[0] != output_width ||
        (have_bias && bias.shape[0] != output_width)) {
        PyErr_SetString(PyExc_ValueError, "linear buffer shapes do not agree");
        PyBuffer_Release(&output);
        if (have_bias) {
            PyBuffer_Release(&bias);
        }
        PyBuffer_Release(&weight);
        PyBuffer_Release(&value);
        return NULL;
    }

    const float *value_data = (const float *)value.buf;
    const float *weight_data = (const float *)weight.buf;
    const float *bias_data = have_bias ? (const float *)bias.buf : NULL;
    float *output_data = (float *)output.buf;
    float left[MAX_INPUT_WIDTH];
    float right[MAX_INPUT_WIDTH];

    for (Py_ssize_t row = 0; row < output_width; ++row) {
        const float *weight_row = weight_data + row * input_width;
        for (Py_ssize_t column = 0; column < input_width; ++column) {
            left[column] = weight_row[column] * value_data[column];
        }
        float *source = left;
        float *destination = right;
        Py_ssize_t active_width = input_width;
        while (active_width > 1) {
            Py_ssize_t pair_count = active_width / 2;
            for (Py_ssize_t pair = 0; pair < pair_count; ++pair) {
                destination[pair] = source[2 * pair] + source[2 * pair + 1];
            }
            if (active_width % 2) {
                destination[pair_count] = source[active_width - 1];
            }
            active_width = pair_count + active_width % 2;
            float *temporary = source;
            source = destination;
            destination = temporary;
        }
        float result = source[0];
        output_data[row] = have_bias ? result + bias_data[row] : result;
    }

    PyBuffer_Release(&output);
    if (have_bias) {
        PyBuffer_Release(&bias);
    }
    PyBuffer_Release(&weight);
    PyBuffer_Release(&value);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {
        "linear_into",
        linear_into,
        METH_VARARGS,
        "Evaluate an exact float32 pairwise linear map."
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_airhockey_pairwise_float32",
    "Exact native pairwise float32 reductions for structured inference.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__airhockey_pairwise_float32(void) {
    return PyModule_Create(&module);
}
