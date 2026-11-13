"""Check that source identity preserves literals and preprocessing semantics."""
import ctypes.util
import importlib.util
import os
from pathlib import Path
import unittest

TOOL = Path(__file__).with_name("compare_source_identity.py")


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "The source identity comparator is missing")
        spec = importlib.util.spec_from_file_location("source_identity", TOOL)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        library = (os.environ.get("ACV_LIBCLANG") or ctypes.util.find_library("clang-10")
                   or ctypes.util.find_library("clang"))
        self.assertIsNotNone(library, "A real libclang tokenizer is required")
        self.tokenizer = self.module.ClangTokenizer(library)

    def compare(self, old, new):
        return self.module.compare_sources(old, new, self.tokenizer)

    def test_indentation_comments_and_expression_spacing_preserve_identity(self):
        result = self.compare('int f(){return 1 + 2;}\n',
                              'int f()\n{\n    // formatting\n    return 1+2;\n}\n')
        self.assertEqual(result["classification"], "TOKEN_AND_DIRECTIVE_EQUIVALENT")
        self.assertFalse(result["runtime_verified"])

    def test_string_whitespace_and_raw_string_content_are_not_normalized(self):
        for old, new in [('const char *x = "a b";', 'const char *x = "ab";'),
                         ('auto x=R"tag(a b)tag";', 'auto x=R"tag(ab)tag";')]:
            with self.subTest(old=old):
                self.assertEqual(self.compare(old, new)["classification"], "TOKEN_CHANGE")

    def test_operator_token_boundary_is_not_whitespace_normalized(self):
        result = self.compare('int f(){return a + +b;}', 'int f(){return a ++ b;}')
        self.assertEqual(result["classification"], "TOKEN_CHANGE")

    def test_macro_function_adjacency_changes_identity(self):
        result = self.compare('#define F(x) (x)\n', '#define F (x) (x)\n')
        self.assertFalse(result["directives_equal"])
        self.assertNotEqual(result["classification"], "TOKEN_AND_DIRECTIVE_EQUIVALENT")

    def test_macro_stringification_preserves_argument_token_spacing(self):
        result = self.compare('#define S(x) #x\nauto v=S(a+b);\n',
                              '#define S(x) #x\nauto v=S(a + b);\n')
        self.assertEqual(result["classification"], "PREPROCESSING_SENSITIVE_REQUIRES_REBUILD")

    def test_macro_continuation_removal_changes_directive_boundaries(self):
        result = self.compare('#define F(x) \\\n ((x)+1)\nint y=F(2);\n',
                              '#define F(x)\n ((x)+1)\nint y=F(2);\n')
        self.assertFalse(result["directives_equal"])

    def test_conditional_compilation_and_include_order_are_not_ignored(self):
        for old, new in [('#if A\nint x;\n#endif\n', '#if B\nint x;\n#endif\n'),
                         ('#include "a.h"\n#include "b.h"\n', '#include "b.h"\n#include "a.h"\n')]:
            with self.subTest(old=old):
                result = self.compare(old, new)
                self.assertFalse(result["directives_equal"])
                self.assertEqual(result["classification"], "DIRECTIVE_CHANGE_REQUIRES_REBUILD")

    def test_directive_moved_across_code_changes_preprocessing_identity(self):
        old = '#define FLAG 1\nint x=FLAG;\n#undef FLAG\n'
        new = 'int x=FLAG;\n#define FLAG 1\n#undef FLAG\n'
        result = self.compare(old, new)
        self.assertTrue(result["body_tokens_equal"])
        self.assertFalse(result["directives_equal"])
        self.assertEqual(result["classification"], "DIRECTIVE_CHANGE_REQUIRES_REBUILD")

    def test_line_sensitive_builtin_is_not_declared_semantically_equal(self):
        result = self.compare('int x=__LINE__;\n', '\nint x=__LINE__;\n')
        self.assertEqual(result["classification"], "PREPROCESSING_SENSITIVE_REQUIRES_REBUILD")


if __name__ == "__main__":
    unittest.main()
