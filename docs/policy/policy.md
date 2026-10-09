# Lumi Privacy Policy

**Last updated: September 28, 2026**

Lumi (`@Lumi_a_bot`) is an AI assistant that runs on Telegram. This policy explains what happens to the information you send it.

The policy covers the instance you are talking to, which runs on the private machine of the person who set it up, and that person is responsible for your data. Lumi is an independent project, not affiliated with Telegram or any AI provider.

### What information Lumi processes

Telegram passes the bot your user ID and display name so it can tell who is speaking. Every message you send goes to an AI model to produce a reply.

To keep a conversation coherent, the bot replays recent turns back to the model on each new message. By default, it keeps the twenty most recent turns.

You can also ask the bot to remember a fact. Saved notes live separately from the transcript, get added to the model's instructions on every message, and are capped at 6,000 characters. They stay until you remove them; you can read or edit them yourself.

When you ask the bot to run a command, read or write a file, search the web, or look up library documentation, it sends the relevant text to whichever service performs that action. The results become part of the conversation.

### How your information is used

Four things happen to it: the reply is generated, the conversation stays continuous, the tasks you ask for get carried out, and notes you chose to save get recalled. Lumi does not build advertising profiles and does not sell personal information to anyone.

Because the bot can act on your behalf, it also needs to know which instructions to trust. Only the account ID registered by the person running the instance may run programs, write files, or change stored notes. In a group chat the bot stays quiet unless you mention it, reply to it, or give it a command directly, so it does not answer every message in the room. Anyone else who tries gets a short refusal and nothing else.

### Services that receive your information

Lumi has no fixed set of outside services. Which ones are active depends on how the person running your instance configured it. Your messages, conversation history, or request text may reach:

**An AI language model provider.** Everything you write, the conversation history, and the record of your tool calls go to whichever AI endpoint was configured, whether that is a commercial provider or a model running on a local machine. Your conversation therefore reaches a third party, the same as with any chatbot.

**A web search service.** The search terms you ask about and the results go to whichever search or page-retrieval service is configured. Some work without the operator holding an account.

**A software documentation service.** When enabled, the libraries named in your conversation are looked up against an external documentation index to return current, version-specific results.

Each handles what it receives under its own terms and privacy policy. Ask the person who runs the instance which ones are active for you.

### How long your information is kept

Everything stays on the machine the bot runs on. Nothing is held on a server operated by the project itself.

| Information | Kept until |
|-|-|
| Your messages and the bot's replies, including the record of any commands | You delete the conversation |
| Notes you asked the bot to remember | You delete the notes |
| Technical logs, kept in a small number of rotating files | They are rotated away or deleted |
| Temporary files created while carrying out your requests | You delete them, or later work overwrites them |

The rotating logs can record technical details about the environment the bot runs in. Read them before you share them.

### How your information is protected

Commands that touch the system pass through a safety layer first. It blocks destructive operations: deleting a whole filesystem path, elevating privileges, or altering shell and SSH startup files. Commands stay inside one working directory, the home directory is pinned away from your real one, and environment variables that look like credentials are withheld from anything the bot runs.

Credentials for outside services live in a private file on the machine running the bot. They are never written into the published source code or the sample configuration that ships with it.

A bot that can act on your computer has significant authority. The person running it decides whether that level of access is appropriate for their setup.

### Your rights

You can read and edit the notes the bot keeps, and ask to see the technical logs. Clearing a single conversation removes its transcript while leaving saved notes intact. You can remove saved notes individually or all at once, or have the bot do it for you. To remove all stored data—transcripts, notes, and working files—stop the bot and delete the files.

If you are in a group chat and have a question about data from your own use of the bot, contact the person who runs the instance. They are the ones who can act on it.

### Changes to this policy

This policy is updated when the bot's behavior or the way it handles data changes. The date at the top shows the last revision. The source code is the authoritative account. Where this document and the code disagree, the code is correct.
