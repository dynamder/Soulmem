# Global Instructions

Limit the amount of comments you put in the code to a strict minimum. You should almost never add comments, except sometimes on non-trivial code, function definitions if the arguments aren't self-explanatory, and struct or enum definitions and their fields.You should try your best to make the code self-explanatory by using clear naming and types.

Doc comments are allowed, but be concise about `just` its usage or mechanial explanation.

Do not remove existing comments unless they are directly related to what you are changing.

When working on the crates in the workspace, open `AGENTS.md` directly contained in the directory.

## Referring to code

- Reference a name, not a position: This text outlives the code it describes, so cite a selector, function, class, pref or flag name; name a file without a line, and avoid positional phrasing ("the rule above X").

## Workflow

See the [dev-convention](docs/dev-convention.md)
