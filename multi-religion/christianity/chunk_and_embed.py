"""
chunk_and_embed.py  —  Christianity / Bible (KJV + Sinhala + Tamil)

All three languages are chunked and embedded into a single DB and FAISS index.
- English (en): full topic preambles + synonym expansions for semantic retrieval
- Sinhala (si): plain chunking, no English preambles
- Tamil   (ta): plain chunking, no English preambles
"""

import json
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"

CORPUS_PATH     = DATA_DIR / "bible_raw.json"
CHUNKS_PATH     = DATA_DIR / "chunks.json"
CHUNKS_DB_PATH  = DATA_DIR / "chunks-en-si-ta.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH      = DATA_DIR / "faiss_index-en-si-ta.bin"

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
MODEL_NAME    = "all-MiniLM-L6-v2"

# Source URLs per language
SOURCE_URLS = {
    "en": "https://www.biblegateway.com/passage/?search={book}&version=KJV",
    "si": "https://www.wordproject.org/bibles/si/index.htm",
    "ta": "https://www.wordproject.org/bibles/tm/index.htm",
}

# ────────────────────────────────────────────────────────────────
# Testament / book sets
# ────────────────────────────────────────────────────────────────

OLD_TESTAMENT_BOOKS = {
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi",
}

NEW_TESTAMENT_BOOKS = {
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
}

# ────────────────────────────────────────────────────────────────
# Genre map
# ────────────────────────────────────────────────────────────────

GENRE_MAP: dict[str, str] = {
    "Genesis": "Law", "Exodus": "Law", "Leviticus": "Law",
    "Numbers": "Law", "Deuteronomy": "Law",
    "Joshua": "History", "Judges": "History", "Ruth": "History",
    "1 Samuel": "History", "2 Samuel": "History",
    "1 Kings": "History", "2 Kings": "History",
    "1 Chronicles": "History", "2 Chronicles": "History",
    "Ezra": "History", "Nehemiah": "History", "Esther": "History",
    "Job": "Wisdom", "Psalms": "Wisdom", "Proverbs": "Wisdom",
    "Ecclesiastes": "Wisdom", "Song of Solomon": "Wisdom",
    "Isaiah": "Major Prophets", "Jeremiah": "Major Prophets",
    "Lamentations": "Major Prophets", "Ezekiel": "Major Prophets",
    "Daniel": "Major Prophets",
    "Hosea": "Minor Prophets", "Joel": "Minor Prophets",
    "Amos": "Minor Prophets", "Obadiah": "Minor Prophets",
    "Jonah": "Minor Prophets", "Micah": "Minor Prophets",
    "Nahum": "Minor Prophets", "Habakkuk": "Minor Prophets",
    "Zephaniah": "Minor Prophets", "Haggai": "Minor Prophets",
    "Zechariah": "Minor Prophets", "Malachi": "Minor Prophets",
    "Matthew": "Gospels", "Mark": "Gospels",
    "Luke": "Gospels", "John": "Gospels",
    "Acts": "Acts",
    "Romans": "Pauline Epistles", "1 Corinthians": "Pauline Epistles",
    "2 Corinthians": "Pauline Epistles", "Galatians": "Pauline Epistles",
    "Ephesians": "Pauline Epistles", "Philippians": "Pauline Epistles",
    "Colossians": "Pauline Epistles", "1 Thessalonians": "Pauline Epistles",
    "2 Thessalonians": "Pauline Epistles", "1 Timothy": "Pauline Epistles",
    "2 Timothy": "Pauline Epistles", "Titus": "Pauline Epistles",
    "Philemon": "Pauline Epistles",
    "Hebrews": "General Epistles", "James": "General Epistles",
    "1 Peter": "General Epistles", "2 Peter": "General Epistles",
    "1 John": "General Epistles", "2 John": "General Epistles",
    "3 John": "General Epistles", "Jude": "General Epistles",
    "Revelation": "Apocalyptic",
}

# ────────────────────────────────────────────────────────────────
# TOPIC PREAMBLE MAP  (English only)
# ────────────────────────────────────────────────────────────────

TOPIC_PREAMBLES: list[tuple[str, int, int, str]] = [
    ("Matthew", 1, 2,
     "Birth of Jesus. Nativity. Virgin birth. Wise men. Magi. Star of Bethlehem. "
     "Joseph and Mary. Jesus born in Bethlehem. Christmas story. Incarnation of Christ."),
    ("Matthew", 3, 4,
     "Baptism of Jesus. John the Baptist. Jesus baptized in Jordan river. "
     "Temptation of Jesus. Devil tempts Jesus in the wilderness. Forty days fasting. "
     "Jesus resists Satan. Beginning of Jesus's ministry."),
    ("Matthew", 5, 7,
     "Sermon on the Mount. Beatitudes. Blessed are the poor in spirit. Blessed are the meek. "
     "Blessed are the merciful. Blessed are the peacemakers. Blessed are the pure in heart. "
     "Salt of the earth. Light of the world. Lord's Prayer. Our Father who art in heaven. "
     "Hallowed be thy name. Thy kingdom come. Daily bread. Forgive us our trespasses. "
     "Turn the other cheek. Love your enemies. Do not judge. Golden rule. Do unto others. "
     "Narrow gate. Seek and you shall find. Ask and it shall be given. "
     "Sermon on the mountain. Jesus teaches on the mountain. How to pray. How to live. "
     "Christian ethics. Kingdom of heaven teachings."),
    ("Matthew", 8, 9,
     "Miracles of Jesus. Jesus heals the sick. Healing the leper. Healing the centurion's servant. "
     "Calming the storm. Jesus walks on water. Jesus casts out demons. Raising the dead. "
     "Faith and healing. Power of Jesus."),
    ("Matthew", 13, 13,
     "Parables of Jesus. Parable of the sower. Parable of the mustard seed. "
     "Parable of the wheat and tares. Parable of the hidden treasure. "
     "Parable of the pearl of great price. Kingdom of heaven parables. "
     "Why Jesus speaks in parables. Teaching through stories."),
    ("Matthew", 14, 14,
     "Feeding the five thousand. Jesus feeds the multitude. Five loaves and two fish. "
     "Miracle of multiplication. Jesus walks on water. Peter walks on water. Faith and doubt."),
    ("Matthew", 18, 18,
     "Forgiveness. How many times should I forgive. Seventy times seven. "
     "Parable of the unforgiving servant. Greatest in the kingdom of heaven. "
     "Children and humility. Sheep going astray. Lost sheep."),
    ("Matthew", 19, 20,
     "Marriage and divorce. Jesus on marriage. Children come to Jesus. "
     "Rich young ruler. Camel through eye of a needle. Parable of the workers in the vineyard. "
     "First shall be last. Eternal life. What must I do to be saved."),
    ("Matthew", 21, 23,
     "Triumphal entry into Jerusalem. Palm Sunday. Jesus enters Jerusalem. "
     "Cleansing the temple. Money changers. Den of thieves. "
     "Parable of the two sons. Parable of the tenants. Parable of the wedding banquet. "
     "Greatest commandment. Love God love your neighbour. Woes to the Pharisees. "
     "Render to Caesar. Hypocrisy of religious leaders."),
    ("Matthew", 24, 25,
     "End times. Signs of the end of the age. Second coming of Christ. "
     "Tribulation. False prophets. Parable of the ten virgins. "
     "Parable of the talents. Judgment of the nations. Sheep and goats. "
     "Final judgment. Apocalypse. What will happen at the end. Rapture."),
    ("Matthew", 26, 27,
     "Last Supper. Passover meal. Betrayal of Jesus. Judas betrays Jesus. "
     "Garden of Gethsemane. Jesus prays before arrest. Arrest of Jesus. "
     "Trial of Jesus. Peter denies Jesus. Crucifixion. Death of Jesus on the cross. "
     "Suffering of Christ. Atonement. Why did Jesus die."),
    ("Matthew", 28, 28,
     "Resurrection of Jesus. Jesus rises from the dead. Empty tomb. "
     "Great Commission. Go and make disciples. Baptize all nations. "
     "Jesus appears after resurrection. He is risen."),
    ("Mark", 1, 1,
     "Baptism of Jesus. John the Baptist. Beginning of Jesus ministry. "
     "Temptation in the wilderness. Jesus calls disciples. Healing in the synagogue."),
    ("Mark", 4, 4,
     "Parable of the sower. Why Jesus teaches in parables. Mustard seed. "
     "Kingdom of God parables. Lamp under a bushel."),
    ("Mark", 10, 10,
     "Rich young man. What must I do to inherit eternal life. "
     "Camel through eye of needle. Children and Jesus. Divorce. Marriage. "
     "Servant of all. Greatness in God's kingdom."),
    ("Mark", 14, 15,
     "Last Supper. Gethsemane. Arrest trial crucifixion death of Jesus. "
     "Betrayal by Judas. Peter's denial. Suffering servant. Atonement."),
    ("Mark", 16, 16,
     "Resurrection. Empty tomb. Jesus appears to Mary Magdalene. Great Commission."),
    ("Luke", 1, 2,
     "Birth of Jesus. Nativity. Mary and Joseph. Shepherds and angels. "
     "Glory to God in the highest. Annunciation to Mary. Magnificat. "
     "Zechariah and Elizabeth. Birth of John the Baptist. Christmas story."),
    ("Luke", 6, 6,
     "Sermon on the Plain. Beatitudes in Luke. Blessings and woes. "
     "Love your enemies. Do good to those who hate you. Golden rule. "
     "Judge not. Speck and plank. Good tree bears good fruit."),
    ("Luke", 10, 10,
     "Parable of the Good Samaritan. Who is my neighbour. Love God and neighbour. "
     "Greatest commandment. Mary and Martha. Sitting at Jesus feet."),
    ("Luke", 11, 11,
     "Lord's Prayer in Luke. How to pray. Ask seek knock. "
     "Our Father. Thy kingdom come. Give us this day our daily bread."),
    ("Luke", 15, 15,
     "Parable of the Prodigal Son. Lost son returns. Father forgives son. "
     "Parable of the lost sheep. Parable of the lost coin. "
     "God's joy over repentance. God's forgiveness. Unconditional love of God. "
     "Coming home to God. Repentance and restoration. Prodigal son story."),
    ("Luke", 18, 18,
     "Parable of the Pharisee and tax collector. Humility in prayer. "
     "Parable of the persistent widow. Pray and not give up. "
     "Children come to Jesus. Faith like a child."),
    ("Luke", 19, 19,
     "Zacchaeus the tax collector. Jesus and Zacchaeus. Salvation comes to his house. "
     "Triumphal entry. Palm Sunday. Jesus weeps over Jerusalem."),
    ("Luke", 22, 23,
     "Last Supper. Betrayal arrest trial crucifixion. "
     "Jesus on the cross. Father forgive them. Today you will be with me in paradise. "
     "Death of Jesus. Why Jesus suffered and died. Atonement."),
    ("Luke", 24, 24,
     "Resurrection. Road to Emmaus. Jesus appears to disciples. "
     "He is risen. Ascension of Jesus."),
    ("John", 1, 1,
     "In the beginning was the Word. Word became flesh. "
     "Jesus as the Word of God. Logos. Incarnation. Light of the world. "
     "Prologue of John. Deity of Christ. Jesus is God."),
    ("John", 3, 3,
     "Born again. Nicodemus. You must be born again. Born of the Spirit. "
     "For God so loved the world. God's love for humanity. "
     "Eternal life through faith. Salvation. How to be saved. "
     "God gave his only Son. Whoever believes shall not perish."),
    ("John", 4, 4,
     "Woman at the well. Samaritan woman. Living water. "
     "True worship. Worship in spirit and truth. Jesus knows our hearts. "
     "God is spirit. Thirst for God."),
    ("John", 6, 6,
     "Bread of life. I am the bread of life. Feeding five thousand. "
     "Jesus walks on water. Spiritual nourishment. "
     "Whoever comes to me will never hunger. Eternal bread."),
    ("John", 8, 8,
     "Woman caught in adultery. Let him cast the first stone. "
     "I am the light of the world. Truth shall set you free. "
     "Freedom from sin. Forgiveness and second chances."),
    ("John", 10, 10,
     "Good Shepherd. I am the good shepherd. Shepherd lays down his life. "
     "My sheep know my voice. Abundant life. Gate of the sheep. "
     "Jesus protects his followers. God's care for us."),
    ("John", 11, 11,
     "Lazarus raised from the dead. I am the resurrection and the life. "
     "Jesus raises Lazarus. Death and resurrection. Mary and Martha mourn. "
     "Jesus weeps. Compassion of Jesus. Power over death."),
    ("John", 13, 13,
     "Jesus washes disciples feet. Servant leadership. Humility. "
     "New commandment. Love one another as I have loved you. "
     "Last Supper. Betrayal of Judas foretold."),
    ("John", 14, 17,
     "I am the way the truth and the life. No one comes to the Father except through me. "
     "Do not let your hearts be troubled. Many rooms in my Father's house. "
     "Holy Spirit. Paraclete. Comforter. Spirit of truth. "
     "Peace I leave with you. Abide in me. Vine and branches. "
     "Fruit of the Spirit. Prayer in Jesus name. "
     "High Priestly Prayer. Jesus prays for his disciples. Unity of believers. "
     "Jesus intercedes for us. Eternal life is knowing God."),
    ("John", 18, 19,
     "Arrest of Jesus. Trial before Pilate. Crucifixion. "
     "It is finished. Death of Jesus. Atonement. Crown of thorns. "
     "Why did Jesus have to die. Sacrifice of Christ."),
    ("John", 20, 21,
     "Resurrection. Thomas doubts. Blessed are those who have not seen yet believe. "
     "Jesus appears after resurrection. Peace be with you. "
     "Feed my sheep. Peter restored. Great Commission in John."),
    ("Acts", 1, 2,
     "Ascension of Jesus. Pentecost. Coming of the Holy Spirit. "
     "Tongues of fire. Peter's first sermon. Three thousand baptized. "
     "Early church. Birth of the church. Community of believers. "
     "They devoted themselves to the apostles teaching."),
    ("Acts", 9, 9,
     "Conversion of Paul. Saul on the road to Damascus. "
     "Blinding light. Paul meets Jesus. Paul becomes a Christian. "
     "Transformation through Christ."),
    ("Acts", 16, 16,
     "Lydia converts. Jailer converts. Paul and Silas in prison. "
     "Praise in suffering. Earthquake frees Paul. What must I do to be saved. "
     "Believe in the Lord Jesus and you will be saved."),
    ("Romans", 1, 3,
     "All have sinned. Sin and its consequences. Wrath of God. "
     "No one is righteous. Fall of humanity. Need for salvation. "
     "Justification by faith. Righteousness of God."),
    ("Romans", 4, 5,
     "Faith of Abraham. Justified by faith not works. "
     "Grace through faith. Peace with God. Hope does not disappoint. "
     "God demonstrates his love. While we were sinners Christ died for us. "
     "Atonement. Reconciliation with God."),
    ("Romans", 6, 6,
     "Dead to sin alive in Christ. Baptism and new life. "
     "Slave to righteousness. Freedom from sin. "
     "Wages of sin is death. Gift of God is eternal life."),
    ("Romans", 8, 8,
     "No condemnation in Christ Jesus. Life in the Spirit. "
     "Spirit of adoption. We are children of God. "
     "All things work together for good. "
     "Nothing can separate us from the love of God. "
     "Romans 8. God's love is unconditional. Assurance of salvation."),
    ("Romans", 10, 10,
     "Salvation is for everyone. Confess with your mouth Jesus is Lord. "
     "Believe in your heart God raised him from the dead. "
     "Everyone who calls on the name of the Lord will be saved. "
     "How to be saved. Salvation prayer. Faith and confession."),
    ("Romans", 12, 12,
     "Living sacrifice. Offer your body as a living sacrifice. "
     "Renewing of the mind. Do not conform to the world. "
     "Different spiritual gifts. Body of Christ. "
     "Love must be sincere. Bless those who persecute you. "
     "Do not repay evil with evil. Overcome evil with good."),
    ("1 Corinthians", 12, 12,
     "Spiritual gifts. Gifts of the Holy Spirit. "
     "Body of Christ. Different parts one body. "
     "Gift of prophecy healing tongues wisdom knowledge faith. "
     "Unity in diversity."),
    ("1 Corinthians", 13, 13,
     "Love chapter. What is love. Love is patient love is kind. "
     "Love does not envy does not boast. Love never fails. "
     "Faith hope and love. Greatest of these is love. "
     "Christian love. Agape love. If I have not love I am nothing. "
     "1 Corinthians 13. Definition of love."),
    ("1 Corinthians", 15, 15,
     "Resurrection of the dead. Christ raised from the dead. "
     "Death is swallowed up in victory. Spiritual body. "
     "Resurrection chapter. What happens after death. "
     "If Christ has not been raised your faith is futile. "
     "Last enemy destroyed is death."),
    ("2 Corinthians", 5, 5,
     "New creation in Christ. Old has gone new has come. "
     "Ministry of reconciliation. Ambassador for Christ. "
     "God was reconciling the world to himself. "
     "What does it mean to be a Christian."),
    ("2 Corinthians", 12, 12,
     "Thorn in the flesh. Paul's weakness. "
     "My grace is sufficient for you. "
     "Power is made perfect in weakness. "
     "Strength in suffering. God's grace in hard times."),
    ("Galatians", 5, 5,
     "Fruit of the Spirit. Love joy peace patience kindness goodness "
     "faithfulness gentleness self-control. "
     "Freedom in Christ. Do not use freedom to indulge the flesh. "
     "Walk by the Spirit. Living in the Spirit."),
    ("Ephesians", 2, 2,
     "Saved by grace through faith not by works. "
     "God's gift of salvation. Dead in sin made alive in Christ. "
     "Grace and mercy of God. Ephesians 2. How are we saved."),
    ("Ephesians", 4, 6,
     "Unity of the body. Spiritual maturity. "
     "Armor of God. Put on the full armor of God. "
     "Belt of truth breastplate of righteousness shoes of peace "
     "shield of faith helmet of salvation sword of the Spirit. "
     "Christian warfare. Spiritual battle. "
     "Wives and husbands. Children obey your parents. "
     "Masters and servants. Christian household."),
    ("Philippians", 4, 4,
     "I can do all things through Christ who strengthens me. "
     "Peace of God that passes all understanding. "
     "Do not be anxious about anything. Pray about everything. "
     "Think about whatever is true noble right pure lovely admirable. "
     "Contentment in all circumstances. Joy in suffering."),
    ("Colossians", 3, 3,
     "Set your minds on things above. Put to death earthly nature. "
     "Put on compassion kindness humility gentleness patience. "
     "Forgive as the Lord forgave you. "
     "Whatever you do do it for the Lord. Christian living."),
    ("Hebrews", 11, 11,
     "Hall of faith. Faith chapter. By faith Abel Noah Abraham Sarah Moses. "
     "Faith is confidence in what we hope for. "
     "Without faith it is impossible to please God. "
     "Heroes of faith. Cloud of witnesses."),
    ("Hebrews", 12, 12,
     "Run the race with perseverance. Cloud of witnesses. "
     "Fix your eyes on Jesus. Discipline of the Lord. "
     "Endure hardship as discipline. God disciplines those he loves."),
    ("James", 1, 2,
     "Trials and temptations. Count it joy when you face trials. "
     "Perseverance. Wisdom from God. Ask God for wisdom. "
     "Faith without works is dead. Show me your faith by what you do. "
     "Practical Christianity. Living out your faith."),
    ("James", 3, 3,
     "Taming the tongue. Power of words. No man can tame the tongue. "
     "Wisdom from above. Earthly wisdom versus godly wisdom. "
     "Peacemakers. Fruit of righteousness."),
    ("1 John", 1, 2,
     "God is light. Walking in the light. "
     "If we confess our sins he is faithful to forgive. "
     "Jesus is our advocate. Propitiation for our sins. "
     "Do not love the world. Love of the Father. "
     "Abiding in Christ. Keeping his commandments."),
    ("1 John", 4, 4,
     "God is love. Whoever lives in love lives in God. "
     "We love because he first loved us. "
     "Perfect love drives out fear. "
     "How to know if we love God. Love one another. "
     "Test the spirits. False prophets."),
    ("Revelation", 1, 3,
     "Seven churches. Letters to the churches. "
     "Alpha and Omega. Beginning and end. "
     "Vision of the risen Christ. Jesus appears to John."),
    ("Revelation", 4, 5,
     "Throne room of heaven. Worship in heaven. "
     "Holy holy holy Lord God Almighty. "
     "Lamb of God. Lion of Judah. Worthy is the Lamb. "
     "Book of Life. Scroll with seven seals."),
    ("Revelation", 20, 22,
     "Final judgment. Great white throne judgment. "
     "New heaven and new earth. New Jerusalem. "
     "God wipes every tear. No more death or mourning or crying or pain. "
     "He who overcomes will inherit all this. "
     "End times. What heaven is like. Eternal life. "
     "Second death. Lake of fire. Millennium. "
     "Come Lord Jesus. Maranatha. Second coming."),
    ("Genesis", 1, 2,
     "Creation. God creates the world. In the beginning God. "
     "Let there be light. Six days of creation. Seventh day rest. Sabbath. "
     "Adam and Eve. Garden of Eden. Origin of humanity. "
     "Image of God. Imago Dei. Why did God create us."),
    ("Genesis", 3, 3,
     "The Fall. Adam and Eve sin. Forbidden fruit. Serpent deceives Eve. "
     "Original sin. Why there is evil and suffering. "
     "God curses the serpent. Expulsion from Eden. "
     "Prophecy of the Messiah. Seed of the woman."),
    ("Genesis", 6, 9,
     "Noah and the flood. Noah's ark. God judges the world. "
     "Rainbow covenant. God's promise to Noah. "
     "New beginning after judgment. God's mercy."),
    ("Genesis", 12, 12,
     "Call of Abraham. Leave your country. I will bless you. "
     "Abrahamic covenant. God's promise to Abraham. Father of many nations. "
     "Faith of Abraham. Obedience to God."),
    ("Genesis", 22, 22,
     "Sacrifice of Isaac. Abraham tests his faith. God provides a lamb. "
     "Jehovah Jireh. God will provide. Foreshadowing of Christ's sacrifice. "
     "Mount Moriah. Faith and obedience."),
    ("Exodus", 3, 4,
     "Moses and the burning bush. I am who I am. "
     "God calls Moses. Ten plagues. Let my people go. "
     "Deliverance from Egypt. God hears the cry of his people."),
    ("Exodus", 20, 20,
     "Ten Commandments. Moses receives the law. "
     "You shall have no other gods. Do not make idols. "
     "Do not take the Lord's name in vain. Remember the Sabbath. "
     "Honor your father and mother. Do not murder. Do not commit adultery. "
     "Do not steal. Do not bear false witness. Do not covet. "
     "Moral law of God. How God wants us to live."),
    ("Psalms", 1, 1,
     "Blessed is the man who does not walk with the wicked. "
     "Delight in the law of the Lord. Tree planted by streams of water. "
     "How to be truly happy. Righteous versus wicked."),
    ("Psalms", 22, 22,
     "My God my God why have you forsaken me. "
     "Suffering of the righteous. Prophetic psalm about crucifixion. "
     "They divide my garments. Pierced my hands and feet. "
     "Messianic psalm. Jesus on the cross."),
    ("Psalms", 23, 23,
     "The Lord is my shepherd. I shall not want. "
     "Green pastures still waters. Valley of the shadow of death. "
     "Fear no evil. Rod and staff comfort me. "
     "Goodness and mercy shall follow me. I will dwell in the house of the Lord. "
     "God's care and guidance. Comfort in suffering. Shepherd psalm."),
    ("Psalms", 51, 51,
     "Have mercy on me O God. Create in me a clean heart. "
     "Wash me and I shall be whiter than snow. "
     "Psalm of repentance. Confession of sin. Forgiveness from God. "
     "Broken and contrite heart. David's repentance after sin with Bathsheba."),
    ("Psalms", 91, 91,
     "He who dwells in the shelter of the Most High. "
     "Under the shadow of the Almighty. He will command his angels. "
     "Protection of God. God protects his people. "
     "No harm will befall you. Refuge in God."),
    ("Psalms", 119, 119,
     "Your word is a lamp to my feet. Light to my path. "
     "Love of God's word. Scripture. Bible. "
     "How sweet are your words to my taste. "
     "Longest psalm. Meditation on God's law."),
    ("Proverbs", 3, 3,
     "Trust in the Lord with all your heart. "
     "Lean not on your own understanding. "
     "In all your ways acknowledge him and he will make your paths straight. "
     "Wisdom from God. How to make good decisions."),
    ("Isaiah", 40, 40,
     "Comfort my people. Those who wait on the Lord shall renew their strength. "
     "Mount up with wings like eagles. Run and not grow weary. "
     "God does not grow tired or weary. Everlasting God. Hope in God."),
    ("Isaiah", 53, 53,
     "Suffering servant. He was pierced for our transgressions. "
     "Crushed for our iniquities. By his wounds we are healed. "
     "We all like sheep have gone astray. Led like a lamb to slaughter. "
     "Messianic prophecy. Jesus foretold. Atonement. Isaiah 53. "
     "Why Jesus had to die. Substitutionary atonement."),
    ("Jeremiah", 29, 29,
     "Plans to prosper you and not to harm you. "
     "Plans to give you hope and a future. "
     "Jeremiah 29:11. God's good plans for us. "
     "Seek me and you will find me when you seek me with all your heart."),
    ("Micah", 6, 6,
     "Act justly love mercy walk humbly with your God. "
     "What does the Lord require of you. "
     "Justice mercy humility. Micah 6:8."),
    ("Malachi", 3, 3,
     "Tithing. Bring the whole tithe into the storehouse. "
     "Test me in this says the Lord. Giving to God. "
     "I the Lord do not change. Faithfulness of God."),
]


def _get_topic_preamble(book: str, chapter: int) -> str:
    for (b, ch_start, ch_end, preamble) in TOPIC_PREAMBLES:
        if b == book and ch_start <= chapter <= ch_end:
            return preamble.strip()
    return ""


# ────────────────────────────────────────────────────────────────
# SYNONYM EXPANSION MAP  (English only)
# ────────────────────────────────────────────────────────────────

SYNONYM_EXPANSIONS: list[tuple[list[str], str]] = [
    (["saved", "salvation", "eternal life", "born again", "justified"],
     "How to be saved. What is salvation. How to get to heaven. "
     "Saved by grace. Born again Christian. Justification by faith."),
    (["pray", "prayer", "our father", "hallowed"],
     "How to pray. What is prayer. Talking to God. "
     "Lord's Prayer. Our Father prayer. Christian prayer."),
    (["love one another", "love your neighbour", "love your enemies",
      "love is patient", "god so loved"],
     "What is Christian love. How to love others. "
     "God's love for us. Agape love. Love commandment."),
    (["forgive", "forgiven", "forgiveness", "trespass"],
     "How to forgive. What does the Bible say about forgiveness. "
     "God forgives sins. Receiving forgiveness. Forgiving others."),
    (["faith", "believe", "trust in the lord", "doubt"],
     "What is faith. How to have faith. "
     "Believing in God. Faith versus doubt. Walking by faith."),
    (["holy spirit", "spirit of god", "comforter", "paraclete",
      "tongues of fire", "pentecost"],
     "Who is the Holy Spirit. What does the Holy Spirit do. "
     "Gift of the Holy Spirit. Filled with the Spirit. "
     "Pentecost. Spirit of truth. Helper."),
    (["grace", "unmerited favour", "gift of god"],
     "What is grace. God's grace. Saved by grace. "
     "Grace versus works. Undeserved favour of God."),
    (["sin", "sinful", "transgression", "iniquity", "all have sinned"],
     "What is sin. What does the Bible say about sin. "
     "Consequences of sin. Confessing sin. Overcoming sin."),
    (["heaven", "eternal life", "kingdom of god", "kingdom of heaven",
      "paradise", "new jerusalem"],
     "What is heaven like. What happens after death. "
     "Going to heaven. Eternal life. Kingdom of God. "
     "Is there life after death."),
    (["suffer", "trial", "tribulation", "affliction", "hardship",
      "thorn in", "weary", "persecuted"],
     "Why do Christians suffer. God in suffering. "
     "Finding strength in hardship. God's comfort in pain. "
     "Suffering for Christ. Enduring trials."),
    (["bapti", "born of water", "buried with him"],
     "What is baptism. Why be baptized. "
     "Baptism and salvation. Baptism meaning. "
     "Water baptism. Spirit baptism."),
    (["commandment", "thou shalt not", "love god", "love thy neighbour",
      "law of moses", "the law"],
     "Ten Commandments. What are the commandments. "
     "God's law. Moral law. How God wants us to live. "
     "Greatest commandment."),
    (["second coming", "rapture", "tribulation", "armageddon",
      "millennium", "antichrist", "beast", "666", "apocalypse",
      "end of the world", "last days", "day of the lord"],
     "Second coming of Christ. End times. What will happen at the end. "
     "Signs of the end times. Rapture. Revelation prophecy."),
    (["heal", "miracle", "leper", "blind", "lame", "raised",
      "water into wine", "walked on water", "five thousand"],
     "Miracles of Jesus. Jesus heals the sick. "
     "Signs and wonders. Power of Jesus. Supernatural acts of Jesus."),
    (["father son holy spirit", "trinity", "three in one",
      "godhead", "in the name of"],
     "What is the Trinity. Father Son and Holy Spirit. "
     "Three persons one God. Deity of Christ. Is Jesus God."),
]


def _get_synonym_expansion(text: str) -> str:
    text_lower = text.lower()
    additions  = []
    for (triggers, expansion) in SYNONYM_EXPANSIONS:
        if any(t in text_lower for t in triggers):
            additions.append(expansion)
    return " ".join(additions)


# ────────────────────────────────────────────────────────────────
# Chunking helper
# ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i: i + size])
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
    return chunks


# ────────────────────────────────────────────────────────────────
# Load corpus
# ────────────────────────────────────────────────────────────────

print("Loading corpus...")
corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(corpus):,} entries")

# Count per language for summary
lang_counts = {}
for item in corpus:
    lang = item.get("language", "en") if isinstance(item, dict) else "en"
    lang_counts[lang] = lang_counts.get(lang, 0) + 1
for lang, count in sorted(lang_counts.items()):
    print(f"  {lang}: {count:,} verses")

# ────────────────────────────────────────────────────────────────
# Chunking — with per-language logic
# ────────────────────────────────────────────────────────────────

print("\nChunking corpus...")
all_chunks = []

for item in tqdm(corpus, desc="Chunking"):
    if isinstance(item, dict):
        raw_text  = item.get("text", "")
        section   = item.get("section") or item.get("id") or "unknown"
        testament = item.get("testament", "Unknown")
        chapter   = int(item.get("chapter", 0))
        language  = item.get("language", "en")
    else:
        raw_text  = str(item)
        section   = "unknown"
        testament = "Unknown"
        chapter   = 0
        language  = "en"

    if not raw_text.strip():
        continue

    if testament == "Unknown":
        if section in OLD_TESTAMENT_BOOKS:
            testament = "Old Testament"
        elif section in NEW_TESTAMENT_BOOKS:
            testament = "New Testament"

    genre      = GENRE_MAP.get(section, "General")
    source_url = SOURCE_URLS.get(language, "").format(book=section.replace(" ", "+"))

    for raw_chunk in chunk_text(raw_text):
        parts = []

        if language == "en":
            # English: full semantic enrichment
            topic = _get_topic_preamble(section, chapter)
            if topic:
                parts.append(topic)
            parts.append(f"Book: {section}. Testament: {testament}. Genre: {genre}.")
            parts.append(raw_chunk)
            synonyms = _get_synonym_expansion(raw_chunk)
            if synonyms:
                parts.append(synonyms)
        else:
            # si / ta: plain text only — no English preambles polluting the embedding space
            parts.append(f"Book: {section}. Testament: {testament}.")
            parts.append(raw_chunk)

        embed_text = " ".join(parts)

        all_chunks.append({
            "text":       raw_chunk,
            "embed_text": embed_text,
            "source":     section,
            "book":       section,
            "testament":  testament,
            "genre":      genre,
            "religion":   "Christianity",
            "language":   language,
            "source_url": source_url,
        })

# Summary per language
chunk_lang_counts = {}
for c in all_chunks:
    lang = c["language"]
    chunk_lang_counts[lang] = chunk_lang_counts.get(lang, 0) + 1
print(f"\nTotal chunks: {len(all_chunks):,}")
for lang, count in sorted(chunk_lang_counts.items()):
    print(f"  {lang}: {count:,} chunks")

# ────────────────────────────────────────────────────────────────
# SQLite
# ────────────────────────────────────────────────────────────────

print(f"\nBuilding {CHUNKS_DB_PATH.name} ...")

if CHUNKS_DB_PATH.exists():
    CHUNKS_DB_PATH.unlink()

con = sqlite3.connect(str(CHUNKS_DB_PATH))
con.executescript("""
    CREATE TABLE chunks (
        id        INTEGER PRIMARY KEY,
        text      TEXT    NOT NULL,
        book      TEXT    NOT NULL DEFAULT '',
        testament TEXT    NOT NULL DEFAULT '',
        genre     TEXT    NOT NULL DEFAULT '',
        source    TEXT    NOT NULL DEFAULT '',
        religion  TEXT    NOT NULL DEFAULT 'Christianity',
        language  TEXT    NOT NULL DEFAULT 'en'
    );

    CREATE INDEX idx_religion_language ON chunks (religion, language);
    CREATE INDEX idx_book              ON chunks (book);
""")

con.executemany(
    "INSERT INTO chunks (id, text, book, testament, genre, source, religion, language) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    [
        (
            i,
            c["text"],
            c.get("book",      ""),
            c.get("testament", ""),
            c.get("genre",     ""),
            c.get("source",    ""),
            c.get("religion",  "Christianity"),
            c.get("language",  "en"),
        )
        for i, c in enumerate(all_chunks)
    ],
)

con.commit()
con.close()

db_mb = CHUNKS_DB_PATH.stat().st_size / 1_048_576
print(f"{CHUNKS_DB_PATH.name} saved  ({db_mb:.1f} MB)")

# chunks.json backup (without embed_text)
CHUNKS_PATH.write_text(
    json.dumps(
        [{k: v for k, v in c.items() if k != "embed_text"} for c in all_chunks],
        indent=2, ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"chunks.json saved")

# ────────────────────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────────────────────

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")
embeddings = model.encode(
    [c["embed_text"] for c in all_chunks],
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

np.save(EMBEDDINGS_PATH, embeddings)
print(f"Embeddings saved  shape: {embeddings.shape}")

# ────────────────────────────────────────────────────────────────
# FAISS index
# ────────────────────────────────────────────────────────────────

print("\nBuilding FAISS index...")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, str(FAISS_PATH))
print(f"FAISS index saved  ({index.ntotal:,} vectors)")

# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("Done. Output files:")
for path, note in [
    (CHUNKS_DB_PATH,  "← runtime (upload to HuggingFace)"),
    (FAISS_PATH,      "← runtime (upload to HuggingFace)"),
    (EMBEDDINGS_PATH, "← local backup"),
    (CHUNKS_PATH,     "← local backup"),
]:
    mb = path.stat().st_size / 1_048_576
    print(f"  {path.name:<28} {mb:>7.1f} MB   {note}")
print("─" * 60)
print(f"\nTotal chunks embedded: {len(all_chunks):,}")
for lang, count in sorted(chunk_lang_counts.items()):
    label = {"en": "English", "si": "Sinhala", "ta": "Tamil"}.get(lang, lang)
    print(f"  {label}: {count:,}")