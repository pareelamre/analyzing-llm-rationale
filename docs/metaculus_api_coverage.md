# Metaculus API coverage for the FutureEval tournament bot

The [official Metaculus API](https://www.metaculus.com/api/) currently documents
eight routes. This inventory distinguishes the five documented routes needed
to forecast and comment from optional account-data and withdrawal utilities.
An additional identity route is used by the bot but is not in that OpenAPI file.

| Route | Tournament purpose | Foresea integration | Verification |
|---|---|---|---|
| `GET /api/posts/` | Discover open questions by tournament, type, and close time | `MetaculusClient.list_open_posts`; paginates, requests `include_descriptions`, authenticates | Unit query/pagination tests; authenticated read-only request |
| `GET /api/posts/{postId}/` | Get full question, scaling, criteria, timestamps, own latest forecast, and available aggregate | `MetaculusClient.get_post` | Unit fixtures; authenticated read-only request |
| `POST /api/questions/forecast/` | Submit a validated binary, multiple-choice, numeric, or discrete forecast | `MetaculusClient.submit_forecast`; guarded write with post-detail readback | Unit payload/readback tests; previous practice submission, not a new tournament submission |
| `GET /api/comments/` | Read staff clarifications and verify own private comment | `get_staff_comments` and `submit_private_comment` readback; uses separate staff or author filters | Unit filter/readback tests; previous practice private-comment readback |
| `POST /api/comments/create/` | Attach required private reasoning with the forecast | `submit_private_comment`; checks author, post, text, privacy, and included forecast | Unit tests; previous practice private-comment readback |
| `POST /api/questions/withdraw/` | Withdraw a forecast | `withdraw_forecast`; requires literal `True` confirmation and post-detail readback; never called by the tournament cycle | Mock transport/confirmation/readback tests; no live withdrawal |
| `GET /api/data/download/` | Export CSV data | `download_data`; requires confirmation, validates scope/filters, streams with a 64 MiB cap, returns ZIP bytes without writing files | Mock transport/content/size tests; account permission not verified |
| `POST /api/data/email/` | Email a data export | `schedule_data_email`; validates scope/filters and requires explicit confirmation | Mock transport/confirmation tests; no live email or permission verification |
| `GET /api/users/me/` | Verify token belongs to the expected bot | `MetaculusClient.current_user`; fails closed on mismatch | Unit tests and authenticated account checks; route is outside the published OpenAPI file |
| `GET /api/comments/{id}/` | Retrieve full text of an old archived comment | `MetaculusClient.get_comment`; requires expected author ID and is not used in the normal submission path | Mock readback test; route is described in the FutureEval resources but absent from the OpenAPI file |

The [FutureEval bot resources](https://www.metaculus.com/notebooks/38928/bot-tournament-resources-page/)
say tournament bots must leave private comments. They also describe a separate
`GET /api/comments/{id}/` endpoint for retrieving old archived comments; the
normal submission path verifies a new comment immediately, so it does not call
that archival route. The resource page limits archival reads to eight calls per
ten seconds; callers must respect that limit. The published API notes that data use for model development
is subject to Metaculus's terms and any required permission; endpoint access
alone is not a grant for unrestricted pastcasting or training.
