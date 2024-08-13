#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import os
import re
import sys
import traceback

if sys.version_info >= (3, 2):
    from functools import lru_cache
    from html import escape as html_escape

else:
    from xml.sax.saxutils import escape as _html_escape

    def lru_cache(*args, **kwargs):
        return lambda f: f

    HTML_ESCAPE_TABLE = {
        "'": "&apos;",
        '"': "&quot;",
    }

    def html_escape(text):
        """
        Replace special characters '"', "'", "&", "<" and ">" to make text
        HTML-safe.
        """
        return _html_escape(text, HTML_ESCAPE_TABLE)


__all__ = [
    "OPEN_TAGS",
    "CLOSE_TAGS",
    "CAPTURE_BLOCKS",
    "CAPTURE_EXPRESSION",
    "CAPTURE_REGEX",
    "END_BLOCK_EXPRESSION_REGEX",
    "BLOCK_EXPRESSION_REGEX",
    "TEMPLATE_PATH_PREFIX",
    "WHITESPACE_BYTES",
    "SCRIPT_PATH",
    "compile_template",
    "template_traceback",
    "template_to_function",
    "html_escape",
]

OPEN_TAGS = [
    r"(?:{{|\s*{{-) ",
    r"(?:{%|\s*{%-) ",
    r"(?:{=|\s*{=-) ",
    r"(?:{#|\s*{#-) ",
]
CLOSE_TAGS = [
    r" (?:}}|-}}\s*)",
    r" (?:%}|-%}\s*)",
    r" (?:=}|-=}\s*)",
    r" (?:#}|-#}\s*)",
]

CAPTURE_BLOCKS = (opn + ".*?" + cls for opn, cls in zip(OPEN_TAGS, CLOSE_TAGS))
CAPTURE_EXPRESSION = r"(.*?)(%s)(.*?)|(.+?)\Z" % ("|".join(CAPTURE_BLOCKS),)
CAPTURE_REGEX = re.compile(CAPTURE_EXPRESSION.encode("utf-8"), re.MULTILINE | re.DOTALL)

END_BLOCK_EXPRESSION_REGEX = re.compile("end(for|while|if|with|try)$")
BLOCK_EXPRESSION_REGEX = re.compile(r"(for|while|(el)?if|with)\s|(try|else|finally)\s*:?|except(\s*:|\s)")

TEMPLATE_PATH_PREFIX = "/._/python-templates/"
WHITESPACE_BYTES = frozenset(b" \t\n\r\x0b\x0c") | {32, 8, 9, 10, 11, 12, 13}

SCRIPT_PATH = os.path.abspath(__file__)


@lru_cache()
def compile_template(path):
    """
    Convert template located at `path` to Python code object. On Python
    versions 3.2 and up, calls to this function are cached with
    functools.lru_cache.
    """
    with open(path, "rb") as iostream:
        template_source = iostream.read()

    block_counts = collections.defaultdict(int)
    depth = 0
    span_map = dict()
    python_source = [
        # As the template is parsed, a dictionary is generated that maps lines
        # of the transpiled source to lines of the template and byte offsets
        # where interpolated blocks start.
        "<Reserved for template offset table.>",
    ]

    def add_line(text):
        """
        Helper function for adding lines to generated Python script.
        """
        python_source.append(" " * depth + text)

    # This could be made more efficient by complicating the regular expressions
    # and using named capture groups to avoid needlessly modifying strings and
    # checking characters, but I don't think the additional complexity is worth
    # it right now.
    for match in CAPTURE_REGEX.finditer(template_source):
        before, raw_block, after, tail = match.groups()
        if raw_block:
            first_bracket_offset = 0
            last_bracket_position = None
            if raw_block[0] in WHITESPACE_BYTES or raw_block[-1] in WHITESPACE_BYTES:
                first_bracket_offset = raw_block.index(b"{")
                raw_block.rindex(b"}")
                block = raw_block.strip()
            else:
                block = raw_block

            # Comment block
            if block[:1] == b"#":
                add_line("_template_output.extend((%r, %r))" % (before, after))
                continue

            # Executable block
            if before:
                before = before.decode("utf-8")
                add_line("_template_output.append(" + repr(before) + ")")

            contents = block[3:-3].replace(b"\n", b" ").strip().decode("utf-8")

            # Statement block
            if block[1:2] == b"%":
                if BLOCK_EXPRESSION_REGEX.match(contents):
                    if not contents.endswith(":"):
                        contents += ":"
                    if contents.startswith(("elif", "else", "except", "finally")):
                        depth -= 1
                    else:
                        block_name = contents.split()[0]
                        block_counts[block_name] += 1
                    add_line(contents)
                    depth += 1
                elif END_BLOCK_EXPRESSION_REGEX.match(contents):
                    block_counts[contents[3:]] -= 1
                    depth -= 1
                else:
                    add_line(contents)

            # Output block
            else:
                # Double parentheses ensures that something like "{{ x = 1 }}"
                # produces a less confusing error:
                #
                #     >>> str((x = 1))
                #       File "<stdin>", line 1
                #         str((x=1))
                #               ^
                #     SyntaxError: invalid syntax
                #     >>> str(x=1)
                #     Traceback (most recent call last):
                #       File "<stdin>", line 1, in <module>
                #     TypeError: 'x' is an invalid keyword argument for ...
                #
                contents = "str((" + contents + "))"
                if block[1:2] == b"{":
                    contents = "_template_escaper(" + contents + ")"
                add_line("_template_output.append(" + contents + ")")

            if after:
                after = after.decode("utf-8")
                add_line("_template_output.append(" + repr(after) + ")")

            # Incremental counting of line numbers would probably be more
            # efficient, but bytes.count is implemented in C, and I don't see
            # this becoming a bottleneck any time soon considering all string
            # manipulation done by the transpiler.
            block_start, block_end = match.span(2)
            block_start += first_bracket_offset
            block_end = last_bracket_position or block_end
            width = block_end - block_start
            lineno = template_source.count(b"\n", None, block_start) + 1
            span_map[len(python_source)] = (lineno, block_start, width)

        else:
            tail = tail.decode("utf-8")
            add_line("_template_output.append(" + repr(tail) + ")")

    if depth:
        messages = list()
        text = "the number of %ss is %s than the number of %ss by %d"
        for block, count in block_counts.items():
            if not count:
                continue
            difference = "less" if count < 0 else "greater"
            message = text % (block, difference, "end" + block, abs(count))
            messages.append(message)

        all_messages = ", and ".join(messages).replace("t", "T", 1) + "."
        raise SyntaxError(all_messages)

    python_source[0] = "_template_span_map[%r] = %r" % (path, span_map)
    script = "\n".join(python_source)

    try:
        return compile(script, TEMPLATE_PATH_PREFIX + path, "exec")
    except SyntaxError as error:
        e_lineno = error.lineno
        error.filename = path
        error.offset = -1
        while e_lineno > 0:
            # It's possible that a given line number isn't in the span map, so
            # the line number in the exception is decremented until a line
            # that's actually in the map is found.
            if e_lineno in span_map:
                error.lineno, true_offset, _ = span_map[e_lineno]
                nl = template_source.index(b"\n", true_offset)
                nl = None if nl < 0 else nl
                error.text = template_source[true_offset:nl].decode("utf-8")
                break
            else:
                e_lineno -= 1
        else:
            error.lineno = -1
        raise


def template_to_function(path, escaper=html_escape):
    """
    Convert template into a callable function. By default, the template output
    will be made HTML-safe, but the content escape method can be changed by
    setting the `escaper` argument.

    The resulting function action can be called using two different conventions
    to pass state into the template. One way to pass state into the function is
    to provide variable names and values as keyword arguments to the function:

    >>> render_inbox = template_to_function("inbox.html")
    >>> html = render_inbox(title="Inbox", email="j.doe@example.com")

    Once execution is finished, any state defined only within the template is
    lost. Alternatively, the template function can be called with a dictionary
    as an argument:

    >>> render_inbox = template_to_function("inbox.html")
    >>> variables = {
    ...     "title": "Inbox",
    ...     "email": "j.doe@example.com",
    ... }
    >>> html = render_inbox(variables)

    Within the template, all members of the dictionary will be accessible as
    variables. When template execution is finished, all of the template's state
    will be preserved in the dictionary; any modified or newly defined
    variables will be reflected in dictionary. Continuing from the example
    above, if the template contained a statement like "{% name = ... %}", the
    dictionary would contain a new key after the template was rendered:

    >>> variables
    {'name': 'Jess Doe', 'email': 'j.doe@example.com', 'title': 'Home Page'}

    To avoid conflicting with definitions used internally by the template
    system, no user-defined variable names may start with "_template_". If an
    error is raised during template execution, the dictionary may contain
    internal variables starting with this prefix.
    """
    abspath = os.path.abspath(path)
    compiled_template = compile_template(abspath)
    template_directory = os.path.dirname(abspath)

    def function(_template_symbol_dictionary=None, **symbols):
        if _template_symbol_dictionary is not None and symbols:
            raise ValueError("Cannot specify _template_symbol_dictionary when using " "keyword arguments as template variables.")
        elif _template_symbol_dictionary:
            symbols = _template_symbol_dictionary

        symbols["_template_escaper"] = escaper

        if "_template_output" in symbols:
            is_include_call = True

        else:

            def include(path, raw=False, escaper=None):
                """
                Incorporate `path` into template output. If `raw` is `False`,
                the file will be parsed as a template and executed, but if
                `raw` is `True`, the contents of the file will be incorporated
                into the output verbatim. Normally, the escape function of the
                calling template will be used to escape the included template,
                but this can be overridden by setting `escaper`. If `path` is a
                relative path, it will be interpreted as being relative to the
                directory of the calling template. Note that this function does
                not return the included data.
                """
                path = os.path.join(template_directory, path)
                if raw:
                    if escaper:
                        raise ValueError("Cannot set escaper when raw=False.")
                    with open(path) as iostream:
                        contents = iostream.read()
                    symbols["_template_output"].append(contents)
                else:
                    my_escaper = symbols["_template_escaper"]
                    if escaper is None:
                        escaper = my_escaper
                    try:
                        template_to_function(path, escaper=escaper)(symbols)
                    finally:
                        symbols["_template_escaper"] = my_escaper

            def raw_include(path):
                """
                Helper function to call `include` with `raw=True`; the
                following two expressions are equivalent:

                >>> raw_include("file.txt")
                >>> include("file.txt", raw=True)
                """
                include(path, raw=True)

            def defined(name):
                """
                Return boolean value indicating whether or not a variable is
                defined. The `name` is given as a string.
                """
                return name in symbols

            is_include_call = False
            symbols.update(
                {
                    "_template_output": list(),
                    "_template_span_map": dict(),
                    "defined": defined,
                    "include": include,
                    "raw_include": raw_include,
                }
            )

        try:
            exec(compiled_template, symbols)
            output = "".join(symbols["_template_output"])
            if not is_include_call:
                del symbols["_template_span_map"]
            return output
        finally:
            if not is_include_call:
                del symbols["_template_output"]
                del symbols["defined"]
                del symbols["include"]
                del symbols["raw_include"]

    return function


def template_traceback(templates_only=False):
    """
    An exception raised inside of a template will produce a traceback that can
    be hard to follow. When this function is called within an
    exception-handling block, it returns a sanitized traceback in the form of a
    string that correctly maps locations in the stack to the corresponding
    locations in the templates.

    This is what a standard stack trace looks like when an exception is raised
    within a template:

        Traceback (most recent call last):
          File "app.py", line 412, in <module>
            main()
          File "app.py", line 253, in main
            print(admin_page())
          File ".../tipyte:.py", line 376, in function
            exec(compiled_template, dict(), symbols)
          File "/._/python-templates/.../admin.html", line 3, in <module>
            <div class="container-fluid">
          File ".../tipyte.py", line 345, in include
            template_to_function(path, escaper=escaper)(symbols)
          File ".../tipyte.py", line 376, in function
            exec(compiled_template, dict(), symbols)
          File "/._/python-templates/.../user-list.html", line 3, in <module>
            <ul>
        NameError: name 'query' is not defined

    This is what the sanitized stack trace returned by this function looks
    like:

        Traceback (most recent call last):
          File "app.py", line 412, in <module>
            main()
          File "app.py", line 253, in main
            print(admin_page())
          File ".../admin.html", line 5, in <module>
            {% include("user-list.html") %}
          File ".../user-list.html", line 4, in <module>
            {% for username, country, status in query("SELECT * FROM Users") %}
        NameError: name 'query' is not defined

    Do not use this function when handling a `SyntaxError`. Any syntax errors
    generated from within a template will already have been modified to include
    all the information needed to easily determine where the syntax error is.
    When the `templates_only` option is set, the only files that will be shown
    in the traceback are templates; lines from pure-Python files will be
    elided. If `templates_only` were set, the following traceback would be
    returned lieu of the one above:

        Traceback (most recent call last):
          File ".../admin.html", line 5, in <module>
            {% include("user-list.html") %}
          File ".../user-list.html", line 4, in <module>
            {% for username, country, status in query("SELECT * FROM Users") %}
        NameError: name 'query' is not defined

    Example usage:

    >>> render_home_page = template_to_function("home.html")
    ... try:
    ...     render_home_page(date="January 10th, 2016")
    ... except Exception as error:
    ...     if not isinstance(error, SyntaxError):
    ...         error.template_traceback = template_traceback()
    ...     raise
    """
    _, error, trace = sys.exc_info()

    frames = list()
    for frame in traceback.extract_tb(trace):
        path, lineno, call, text = frame

        if path.startswith(TEMPLATE_PATH_PREFIX):
            path = path[len(TEMPLATE_PATH_PREFIX) :]
            span_map = trace.tb_frame.f_locals["_template_span_map"][path]
            text = None

            try:
                with open(path) as iostream:
                    real_lineno, start, width = span_map[lineno]
                    iostream.seek(start)
                    text = iostream.read(width).replace("\n", " ").strip()
                    lineno = real_lineno
            except Exception:
                pass

            frames.append((path, lineno, call, text))

        elif not templates_only:
            if not os.path.samefile(path, SCRIPT_PATH):
                frames.append(frame)

        trace = trace.tb_next

    if frames:
        prefix = "Traceback (most recent call last):\n"
        middle = "".join(traceback.format_list(frames))
        suffix = error.__class__.__name__ + ": " + str(error)
        return prefix + middle + suffix
