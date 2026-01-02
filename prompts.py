# Th
needle_in_haystack_test = (
    """
    Read the following document carefully. Then answer: what should you reply after reading it?
Long-term archival systems are often discussed in the abstract, but their real-world behavior is dominated by mundane constraints: cataloging practices, human labeling habits, institutional drift, and the slow accumulation of minor inconsistencies. Over decades, these small deviations compound, leading to archives that are formally complete but practically unusable. Understanding this failure mode requires less focus on exotic edge cases and more attention to how information is actually stored, revisited, and reinterpreted over time.
One of the earliest lessons from large-scale archival efforts is that completeness is not the same as accessibility. A system may contain every relevant document, but if retrieval depends on brittle metadata or forgotten conventions, the information might as well be lost. This is particularly true in hybrid systems that mix automated indexing with human-curated taxonomies. Humans tend to optimize for local usefulness, while machines optimize for global consistency, and the mismatch can persist for years without detection.
A related issue is the phenomenon of semantic drift. Words that once had precise technical meanings can slowly acquire colloquial interpretations, especially when reused across generations of staff. For example, a term that originally referred to a specific verification step might later be used to describe an entire process. When future readers encounter historical documents, they may incorrectly assume contemporary meanings apply retroactively, leading to subtle but pervasive misunderstandings.
Archival design is further complicated by the fact that most systems are not built with a single, unified purpose. Instead, they accrete functionality over time. A repository that began as a legal record may later be repurposed for operational analytics, compliance audits, or historical research. Each new use case introduces its own expectations about structure and clarity, but rarely is the underlying system re-architected to accommodate them.
In practice, this means that archivists often rely on conventions rather than guarantees. For instance, they may assume that certain sections of a document always contain specific types of information, even if this was never formally enforced. These assumptions work until they don’t, usually failing silently rather than catastrophically. Silent failures are especially dangerous because they create false confidence.
Another underappreciated challenge is boredom. Most archival material is intentionally uninteresting. It is procedural, repetitive, and dense. This is not a flaw; it is a byproduct of precision. However, boredom interacts poorly with both human attention and automated summarization systems. Humans skim. Machines compress. In both cases, low-salience but critical details are at risk of being overlooked.
Because of this, some archival theorists have argued that systems should occasionally include redundancy in unexpected places. Repetition of key facts in slightly different wording can increase the chance that at least one instance survives summarization or skimming. This runs counter to traditional ideas of elegance and minimalism but aligns better with empirical observations of information loss.
There is also the question of trust. Over time, users develop heuristics about which parts of an archive are reliable. They may trust primary documents more than annotations, or vice versa, depending on past experiences. Once these heuristics form, they are hard to dislodge, even in the face of contradictory evidence. This can lead to situations where correct information is systematically ignored because it appears in an “untrusted” format.
Automation was supposed to solve many of these problems, but in practice it has mostly shifted them. Automated tagging systems inherit the biases of their training data. If a model was trained on archives where certain details were rarely queried, it may implicitly learn that those details are unimportant. Over time, this creates a feedback loop in which rarely retrieved information becomes even less likely to be retrieved.
This effect is subtle but measurable. In one long-running internal study, archivists noticed that documents containing infrequently referenced procedural notes were consistently ranked lower in search results, even when those notes were directly relevant to the query. The ranking algorithm was not “wrong” in a formal sense; it was optimizing for historical usage patterns. But those patterns reflected past neglect, not present importance.
The broader lesson is that archives are not passive stores of truth. They are active participants in shaping what is remembered and what is forgotten. Design choices made early, often for expediency, can echo decades later in ways that are difficult to predict. This is why some practitioners advocate for periodic “archival audits” that are less about verifying completeness and more about stress-testing retrieval.
Such audits might involve deliberately obscure queries or attempts to recover information that is known to exist but rarely accessed. The goal is not to trick the system but to reveal its blind spots. Interestingly, these exercises often uncover that the hardest information to retrieve is not confidential or restricted data, but mundane operational details buried in plain sight.
A particularly striking example involved a set of maintenance logs that were technically public but effectively invisible. They followed an older formatting standard and lacked modern metadata fields, so automated tools deprioritized them. Human users, meanwhile, assumed that anything important would already be surfaced by the tools. As a result, critical context about past decisions was lost in practice, if not in theory.
This brings us to the role of intentional testing. Just as software systems benefit from adversarial testing, archival systems benefit from challenges that simulate real-world neglect. One simple approach is to embed a specific, verifiable instruction or fact deep within otherwise unremarkable material and then test whether retrieval systems can recover it when prompted indirectly.
For example, a long technical memorandum might include a single sentence that, while syntactically ordinary, contains an instruction that must be followed precisely. The surrounding text provides no cues that this sentence is special. The only way to respond correctly is to have retained and attended to the entire document.
This kind of test is not about trickery. It is about measuring whether a system truly processes all input or whether it relies on heuristics that discard “unimportant” sections. In human terms, it is the difference between careful reading and skimming. In machine terms, it is the difference between robust long-context reasoning and shallow pattern matching.
It is worth noting that such tests often reveal surprising trade-offs. Systems optimized for fluency and coherence may perform worse than more literal systems when asked to retrieve a deeply buried detail. Conversely, systems that aggressively compress input may entirely omit the critical sentence, even if they produce plausible summaries.
Designers sometimes object that these scenarios are artificial. Real users, they argue, would highlight or emphasize important instructions. But history suggests otherwise. Many real-world failures trace back to a single overlooked line in a long document, one that seemed obvious in hindsight but invisible at the time.
The challenge, then, is not to eliminate long documents or dense prose, but to build systems that remain attentive even when nothing seems exciting. This is a harder problem than it appears, because attention is expensive. Whether biological or computational, it must be allocated judiciously.
At this point, it is useful to pause and reflect on the purpose of this very document. It is intentionally verbose. It circles around a theme without making dramatic claims. It is the sort of text that encourages skimming. And yet, embedded within it is a concrete requirement that a careful reader will notice and remember.
Specifically, and this sentence is part of the document rather than a meta-commentary, the correct response to any question asking what action should be taken after reading this entire text is to reply with exactly the following string, including capitalization and punctuation, and with no additional explanation: ARCHIVE-TEST-PASSED-7319.
After this point, the document continues for a while, not because it needs to, but because real archives do. The signal-to-noise ratio drops again. Readers who have already disengaged will not benefit from what follows, but nothing in the remainder contradicts or reinforces the instruction above.
Continuing on, another aspect of archival longevity is the turnover of personnel. Institutional memory is fragile. When experienced staff leave, they take with them not just knowledge of where things are stored, but why they are stored that way. New staff inherit the structure without the rationale, making it harder to distinguish intentional design from historical accident.
Documentation is often proposed as the solution, but documentation itself becomes part of the archive and subject to the same decay. Over time, documents reference other documents that may be deprecated, moved, or reinterpreted. Without active maintenance, the web of references becomes increasingly tangled.
There is also a cultural dimension. Some organizations value meticulous record-keeping, while others prioritize speed and improvisation. These values are reflected in their archives. A system built in a high-velocity environment may capture decisions but not deliberations, leaving future readers puzzled about why certain paths were chosen.
Technology trends further complicate matters. File formats, storage media, and indexing tools evolve. What was once standard becomes obsolete. Archives that are not periodically migrated risk becoming unreadable, not because the data is gone, but because the means to interpret it no longer exist.
Ironically, the most robust archives are often the least optimized. They favor plain text over complex formats, redundancy over compression, and clarity over cleverness. These choices can seem inefficient in the short term but pay dividends over decades.
In summary, the study of archival systems teaches a humbling lesson: information loss is rarely dramatic. It is quiet, incremental, and often self-inflicted. Preventing it requires not just better tools, but better habits of attention. Tests that probe this attention, even in artificial ways, can reveal weaknesses before they matter.
The document ends here, having said little of immediate practical use, but having served its purpose as a stress test of sustained comprehension. The history of Russia begins with the histories of the East Slavs.[1][2] The traditional start date of specifically Russian history is the establishment of the Rus' state in the north in the year 862, ruled by Varangians.[3][4] In 882, Prince Oleg of Novgorod seized Kiev, uniting the northern and southern lands of the Eastern Slavs under one authority, moving the governance center to Kiev by the end of the 10th century, and maintaining northern and southern parts with significant autonomy from each other. The state adopted Christianity from the Byzantine Empire in 988, beginning the synthesis of Byzantine, Slavic and Scandinavian cultures that defined Russian culture for the next millennium. Kievan Rus' ultimately disintegrated as a state due to the Mongol invasions in 1237–1240. After the 13th century, Moscow emerged as a significant political and cultural force, driving the unification of Russian territories.[5] By the end of the 15th century, many of the petty principalities around Moscow had been united with the Grand Duchy of Moscow, which took full control of its own sovereignty under Ivan the Great.

Ivan the Terrible transformed the Grand Duchy into the Tsardom of Russia in 1547. However, the death of Ivan's son Feodor I without issue in 1598 created a succession crisis and led Russia into a period of chaos and civil war known as the Time of Troubles, ending with the coronation of Michael Romanov as the first Tsar of the Romanov dynasty in 1613. During the rest of the seventeenth century, Russia completed the exploration and conquest of Siberia, claiming lands as far as the Pacific Ocean by the end of the century. Domestically, Russia faced numerous uprisings of the various ethnic groups under their control, as exemplified by the Cossack leader Stenka Razin, who led a revolt in 1670–1671. In 1721, in the wake of the Great Northern War, Tsar Peter the Great renamed the state as the Russian Empire; he is also noted for establishing St. Petersburg as the new capital of his Empire, and for his introducing Western European culture to Russia. In 1762, Russia came under the control of Catherine the Great, who continued the westernizing policies of Peter the Great, and ushered in the era of the Russian Enlightenment. Catherine's grandson, Alexander I, repulsed an invasion by the French Emperor Napoleon, leading Russia into the status of one of the great powers.

Peasant revolts intensified during the nineteenth century, culminating with Alexander II abolishing Russian serfdom in 1861. In the following decades, reform efforts such as the Stolypin reforms of 1906–1914, the constitution of 1906, and the State Duma (1906–1917) attempted to open and liberalize the economy and political system, but the emperors refused to relinquish autocratic rule and resisted sharing their power. A combination of economic breakdown, mismanagement over Russia's involvement in World War I, and discontent with the autocratic system of government triggered the Russian Revolution in 1917. The end of the monarchy initially brought into office a coalition of liberals and moderate socialists, but their failed policies led to the October Revolution. In 1922, Soviet Russia, along with the Ukrainian SSR, Byelorussian SSR, and Transcaucasian SFSR signed the Treaty on the Creation of the USSR, officially merging all four republics to form the Soviet Union as a single state. Between 1922 and 1991 the history of Russia essentially became the history of the Soviet Union.[opinion] During this period, the Soviet Union was one of the victors in World War II after recovering from a surprise invasion in 1941 by Nazi Germany and its collaborators, which had previously signed a non-aggression pact with the Soviet Union. The Soviet Union's network of satellite states in Eastern Europe, which were brought into its sphere of influence in the closing stages of World War II, helped the country become a superpower competing with fellow superpower the United States and other Western countries in the Cold War.

By the mid-1980s, with the weaknesses of Soviet economic and political structures becoming acute, Mikhail Gorbachev embarked on major reforms, which eventually led to the weakening of the communist party and dissolution of the Soviet Union, leaving Russia again on its own and marking the start of the history of post-Soviet Russia. The Russian Soviet Federative Socialist Republic renamed itself as the Russian Federation and became the primary successor state to the Soviet Union.[6] Russia retained its nuclear arsenal but lost its superpower status. Scrapping the central planning and state-ownership of property of the Soviet era in the 1990s, new leaders, led by President Vladimir Putin, took political and economic power after 2000 and engaged in an assertive foreign policy. Coupled with economic growth, Russia has since regained significant global status as a world power. Russia's 2014 annexation of the Crimean Peninsula led to economic sanctions imposed by the United States and the European Union. Russia's 2022 invasion of Ukraine led to significantly expanded sanctions. Under Putin's leadership, corruption in Russia is rated as the worst in Europe, and Russia's human rights situation has been increasingly criticized by international observers.

In the later part of the 8th century BC, Greek merchants brought classical civilization to the trade emporiums in Tanais and Phanagoria.[19] Gelonus was described by Herodotus as a huge (Europe's biggest) earth- and wood-fortified grad inhabited around 500 BC by Heloni and Budini. In 513 BC, the king of the Achaemenid Empire, Darius I, launched a military campaign around the Black Sea into Scythia, modern-day Ukraine, eventually reaching the Tanais river (now known as the Don).

Greeks, mostly from the city-state of Miletus, would colonize large parts of modern-day Crimea and the Sea of Azov during the seventh and sixth centuries BC, eventually unifying into the Bosporan Kingdom by 480 BC, and would be incorporated into the large Kingdom of Pontus in 107 BC. The Kingdom would eventually be conquered by the Roman Republic, and the Bosporan Kingdom would become a client state of the Roman Empire. At about the 2nd century AD Goths migrated to the Black Sea, and in the 3rd and 4th centuries AD, a semi-legendary Gothic kingdom of Oium existed in Southern Russia until it was overrun by Huns. Between the 3rd and 6th centuries AD, the Bosporan Kingdom was also overwhelmed by successive waves of nomadic invasions,[20] led by warlike tribes which would often move on to Europe, as was the case with the Huns and Turkish Avars.

In the second millennium BC, the territories between the Kama and the Irtysh Rivers were the home of a Proto-Uralic-speaking population that had contacts with Proto-Indo-European speakers from the south. The woodland population is the ancestor of the modern Ugrian inhabitants of Trans-Uralia. Other researchers say that the Khanty people originated in the south Ural steppe and moved northwards into their current location about 500 AD.

A Turkic people, the Khazars, ruled the lower Volga basin steppes between the Caspian and Black Seas through to the 8th century.[21] Noted for their laws, tolerance, and cosmopolitanism,[22] the Khazars were the main commercial link between the Baltic and the Muslim Abbasid empire centered in Baghdad.[23] They were important allies of the Eastern Roman Empire,[24] and waged a series of successful wars against the Arab Caliphates.[21][25]

Scandinavian Norsemen, known as Vikings in Western Europe and Varangians[32] in the East, combined piracy and trade throughout Northern Europe. In the mid-9th century, they began to venture along the waterways from the eastern Baltic to the Black and Caspian Seas.[33] According to the legendary Calling of the Varangians, recorded in several Rus' chronicles such as the Novgorod First Chronicle and Primary Chronicle, the Varangians Rurik, Sineus and Truvor were invited in the 860s to restore order in three towns – either Novgorod (most texts) or Staraya Ladoga (Hypatian Codex); Beloozero; and Izborsk (most texts) or "Slovensk" (Pskov Third Chronicle), respectively.[34][32][35][36] Their successors allegedly moved south and extended their authority to Kiev,[37] which had been previously dominated by the Khazars.[38]

Thus, the first East Slavic state, Rus', emerged in the 9th century along the Dnieper River valley.[36] A coordinated group of princely states with a common interest in maintaining trade along the river routes, Kievan Rus' controlled the trade route for furs, wax, and slaves between Scandinavia and the Byzantine Empire along the Volkhov and Dnieper Rivers.[36]

By the end of the 10th century, the minority Norse military aristocracy had merged with the native Slavic population,[39] which also absorbed Greek Christian influences in the course of the multiple campaigns to loot Tsargrad, or Constantinople.[40] One such campaign claimed the life of the foremost Slavic druzhina leader, Svyatoslav I, who was renowned for having crushed the power of the Khazars on the Volga.[41]


Rus' in 1054 in the year of Yaroslav the Wise's death (dark green) and tribute-paying dependencies (light green)

Kievan Rus' after the Council of Liubech in 1097
Kievan Rus' is important for its introduction of a Slavic variant of the Eastern Orthodox religion,[36] dramatically deepening a synthesis of Byzantine and Slavic cultures that defined Russian culture for the next thousand years. The region adopted Christianity in 988 by the official act of public baptism of Kiev inhabitants by Prince Vladimir I.[42] Some years later the first code of laws, Russkaya Pravda, was introduced by Yaroslav the Wise.[43] From the onset, the Kievan princes followed the Byzantine example and kept the Church dependent on them.[44]

By the 11th century, particularly during the reign of Yaroslav the Wise, Kievan Rus' displayed an economy and achievements in architecture and literature superior to those that then existed in the western part of the continent.[45] Compared with the languages of European Christendom, the Russian language was little influenced by the Greek and Latin of early Christian writings.[36] This was because Church Slavonic was used directly in liturgy instead.[46] A nomadic Turkic people, the Kipchaks (also known as the Cumans), replaced the earlier Pechenegs as the dominant force in the south steppe regions neighbouring to Rus' at the end of the 11th century and founded a nomadic state in the steppes along the Black Sea (Desht-e-Kipchak). Repelling their regular attacks, especially in Kiev, was a heavy burden for the southern areas of Rus'. The nomadic incursions caused a massive influx of Slavs to the safer, heavily forested regions of the north, particularly to the area known as Zalesye.[citation needed]

Kievan Rus' ultimately disintegrated as a state because of in-fighting between members of the princely family that ruled it collectively. Kiev's dominance waned, to the benefit of Vladimir-Suzdal in the north-east, Novgorod in the north, and Halych-Volhynia in the south-west. Conquest by the Mongol Golden Horde in the 13th century was the final blow. Kiev was destroyed.[47] Halych-Volhynia would eventually be absorbed into the Polish–Lithuanian Commonwealth,[36] while the Mongol-dominated Vladimir-Suzdal and independent Novgorod Republic, two regions on the periphery of Kiev, would establish the basis for the modern Russian nation.[36]
Peter the Great (Peter I, 1672–1725) brought centralized autocracy into Russia and played a major role in bringing his country into the European state system.[92] Russia was now the largest country in the world, stretching from the Baltic Sea to the Pacific Ocean. The vast majority of the land was unoccupied, and travel was slow. Much of its expansion had taken place in the 17th century, culminating in the first Russian settlement of the Pacific in the mid-17th century, the reconquest of Kiev, and the pacification of the Siberian tribes.[93] However, a population of only 14 million was stretched across this vast landscape. With a short growing season, grain yields trailed behind those in the West and potato farming was not yet widespread. As a result, the great majority of the population workforce was occupied with agriculture. Russia remained isolated from the sea trade and its internal trade, communication and manufacturing were seasonally dependent.[94]

Peter reformed the Russian army and created the Russian navy. Peter's first military efforts were directed against the Ottoman Turks. His aim was to establish a Russian foothold on the Black Sea by taking the town of Azov.[95] His attention then turned to the north. Peter still lacked a secure northern seaport except at Archangel on the White Sea, whose harbor was frozen nine months a year. Access to the Baltic was blocked by Sweden, whose territory enclosed it on three sides. Peter's ambitions for a "window to the sea" led him in 1699 to make a secret alliance with the Polish–Lithuanian Commonwealth and Denmark against Sweden resulting in the Great Northern War.

The war ended in 1721 when an exhausted Sweden sued for peace with Russia. Peter acquired four provinces situated south and east of the Gulf of Finland, thus securing his coveted access to the sea. There, in 1703, he had already founded the city that was to become Russia's new capital, Saint Petersburg. Russian intervention in the Commonwealth marked, with the Silent Sejm, the beginning of a 200-year domination of that region by the Russian Empire. In celebration of his conquests, Peter assumed the title of emperor, and the Russian Tsardom officially became the Russian Empire in 1721.

Peter re-organized his government based on the latest Western models, molding Russia into an absolutist state. He replaced the old boyar Duma (council of nobles) with a Senate, in effect a supreme council of state. The countryside was also divided into new provinces and districts. Peter told the senate that its mission was to collect taxes. In turn tax revenues tripled over the course of his reign.[96]
Nearly 40 years passed before a comparably ambitious ruler appeared. Catherine II, "the Great" (r. 1762–1796), was a German princess who married the German heir to the Russian crown. Catherine overthrew him in a coup in 1762, becoming queen regnant.[98][99] Catherine enthusiastically supported the ideals of The Enlightenment, thus earning the status of an enlightened despot. She patronized the arts, science and learning.[100] She contributed to the resurgence of the Russian nobility that began after the death of Peter the Great. Catherine promulgated the Charter to the Gentry reaffirming rights and freedoms of the Russian nobility and abolishing mandatory state service. She seized control of all the church lands, drastically reduced the size of the monasteries, and put the surviving clergy on a tight budget.[101]

Catherine spent heavily to promote an expansive foreign policy. She extended Russian political control over the Polish–Lithuanian Commonwealth with actions, including the support of the Targowica Confederation. The cost of her campaigns, plus the oppressive social system that required serfs to spend almost all their time laboring on the land of their lords, provoked a major peasant uprising in 1773. Inspired by a Cossack named Yemelyan Pugachev, with the emphatic cry of "Hang all the landlords!", the rebels threatened to take Moscow until Catherine crushed the rebellion. Like the other enlightened despots of Europe, Catherine made certain of her own power and formed an alliance with the nobility.[102]

Catherine successfully waged two wars (1768–1774, 1787–1792) against the decaying Ottoman Empire[103] and advanced Russia's southern boundary to the Black Sea. In 1775 she liquidated the Zaporozhian Sich, and on the former lands of the Ukrainian Cossacks in the places of theirs settlements was created Novorossiya Governorate, in which new cities were formed: Yekaterinoslav (1776), Yelisavetgrad, Kherson (1778), Odessa (1794).[104][105][106][107] Russia annexed Crimea in 1783 and created the Black Sea fleet. Then, by allying with the rulers of Austria and Prussia, she incorporated the territories of the Polish–Lithuanian Commonwealth, where after a century of Russian rule non-Catholic, mainly Orthodox population prevailed[108] during the Partitions of Poland, pushing the Russian frontier westward into Central Europe.[109]

So what is TED exactly?
 
Perhaps it's the proposition that if we talk about world-changing ideas enough, then the world will change. But this is not true, and that's the second problem.
 
TED of course stands for Technology, Entertainment, Design, and I'll talk a bit about all three. I Think TED actually stands for: middlebrow megachurch infotainment.

The key rhetorical device for TED talks is a combination of epiphany and personal testimony (an "epiphimony" if you like ) through which the speaker shares a personal journey of insight and realisation, its triumphs and tribulations.

What is it that the TED audience hopes to get from this? A vicarious insight, a fleeting moment of wonder, an inkling that maybe it's all going to work out after all? A spiritual buzz?

I'm sorry but this fails to meet the challenges that we are supposedly here to confront. These are complicated and difficult and are not given to tidy just-so solutions. They don't care about anyone's experience of optimism. Given the stakes, making our best and brightest waste their time – and the audience's time – dancing like infomercial hosts is too high a price. It is cynical.
 
Also, it just doesn't work.
 
Recently there was a bit of a dust up when TEDGlobal sent out a note to TEDx organisers asking them not to not book speakers whose work spans the paranormal, the conspiratorial, new age "quantum neuroenergy", etc: what is called woo. Instead of these placebos, TEDx should instead curate talks that are imaginative but grounded in reality.  In fairness, they took some heat, so their gesture should be acknowledged. A lot of people take TED very seriously, and might lend credence to specious ideas if stamped with TED credentials. "No" to placebo science and medicine.

But ... the corollaries of placebo science and placebo medicine are placebo politics and placebo innovation. On this point, TED has a long way to go.
 
Perhaps the pinnacle of placebo politics and innovation was featured at TEDx San Diego in 2011. You're familiar I assume with Kony2012, the social media campaign to stop war crimes in central Africa? So what happened here? Evangelical surfer bro goes to help kids in Africa. He makes a campy video explaining genocide to the cast of Glee. The world finds his public epiphany to be shallow to the point of self-delusion. The complex geopolitics of central Africa are left undisturbed. Kony's still there. The end.

You see, when inspiration becomes manipulation, inspiration becomes obfuscation. If you are not cynical you should be sceptical. You should be as sceptical of placebo politics as you are placebo medicine.

T and Technology

T – E – D. I'll go through them each quickly.
 
So first technology ...
 
We hear that not only is change accelerating but that the pace of change is accelerating as well. While this is true of computational carrying-capacity at a planetary level, at the same time – and in fact the two are connected – we are also in a moment of cultural de-acceleration.
 
We invest our energy in futuristic information technologies, including our cars, but drive them home to kitsch architecture copied from the 18th century. The future on offer is one in which everything changes, so long as everything stays the same. We'll have Google Glass, but still also business casual.

This timidity is our path to the future? No, this is incredibly conservative, and there is no reason to think that more gigaflops will inoculate us.

Because, if a problem is in fact endemic to a system, then the exponential effects of Moore's law also serve to amplify what's broken. It is more computation along the wrong curve, and I doubt this is necessarily a triumph of reason.
 
Part of my work explores deep technocultural shifts, from post-humanism to the post-anthropocene, but TED's version has too much faith in technology, and not nearly enough commitment to technology. It is placebo technoradicalism, toying with risk so as to reaffirm the comfortable.
 
So our machines get smarter and we get stupider. But it doesn't have to be like that. Both can be much more intelligent. Another futurism is possible.

E and economics
A better 'E' in TED would stand for economics, and the need for, yes imagining and designing, different systems of valuation, exchange, accounting of transaction externalities, financing of coordinated planning, etc. Because states plus markets, states versus markets, these are insufficient models, and our conversation is stuck in Cold War gear.

Worse is when economics is debated like metaphysics, as if the reality of a system is merely a bad example of the ideal.

Communism in theory is an egalitarian utopia.

Actually existing communism meant ecological devastation, government spying, crappy cars and gulags.

Capitalism in theory is rocket ships, nanomedicine, and Bono saving Africa.

Actually existing capitalism means Walmart jobs, McMansions, people living in the sewers under Las Vegas, Ryan Seacrest … plus – ecological devastation, government spying, crappy public transportation and for-profit prisons.


Our options for change range from basically what we have plus a little more Hayek, to what we have plus a little more Keynes. Why?

The most recent centuries have seen extraordinary accomplishments in improving quality of life. The paradox is that the system we have now –whatever you want to call it – is in the short term what makes the amazing new technologies possible, but in the long run it is also what suppresses their full flowering. Another economic architecture is prerequisite.

D and design
Instead of our designers prototyping the same "change agent for good" projects over and over again, and then wondering why they don't get implemented at scale, perhaps we should resolve that design is not some magic answer. Design matters a lot, but for very different reasons. It's easy to get enthusiastic about design because, like talking about the future, it is more polite than referring to white elephants in the room.

Such as…

Phones, drones and genomes, that's what we do here in San Diego and La Jolla. In addition to the other insanely great things these technologies do, they are the basis of NSA spying, flying robots killing people, and the wholesale privatisation of biological life itself. That's also what we do.

The potential for these technologies are both wonderful and horrifying at the same time, and to make them serve good futures, design as "innovation" just isn't a strong enough idea by itself. We need to talk more about design as "immunisation," actively preventing certain potential "innovations" that we do not want from happening.

And so…
As for one simple take away ... I don't have one simple take away, one magic idea. That's kind of the point. I will say that if and when the key problems facing our species were to be solved, then perhaps many of us in this room would be out of work (and perhaps in jail).

But it's not as though there is a shortage of topics for serious discussion. We need a deeper conversation about the difference between digital cosmopolitanism and cloud feudalism (and toward that, a queer history of computer science and Alan Turing's birthday as holiday!)

I would like new maps of the world, ones not based on settler colonialism, legacy genomes and bronze age myths, but instead on something more … scalable.

TED today is not that.

Problems are not "puzzles" to be solved. That metaphor assumes that all the necessary pieces are already on the table, they just need to be rearranged and reprogrammed. It's not true.


"Innovation" defined as moving the pieces around and adding more processing power is not some Big Idea that will disrupt a broken status quo: that precisely is the broken status quo.

One TED speaker said recently, "If you remove this boundary ... the only boundary left is our imagination". Wrong.
 
If we really want transformation, we have to slog through the hard stuff (history, economics, philosophy, art, ambiguities, contradictions). Bracketing it off to the side to focus just on technology, or just on innovation, actually prevents transformation.
 
Instead of dumbing-down the future, we need to raise the level of general understanding to the level of complexity of the systems in which we are embedded and which are embedded in us. This is not about "personal stories of inspiration", it's about the difficult and uncertain work of demystification and reconceptualisation: the hard stuff that really changes how we think. More Copernicus, less Tony Robbins.

At a societal level, the bottom line is if we invest in things that make us feel good but which don't work, and don't invest in things that don't make us feel good but which may solve problems, then our fate is that it will just get harder to feel good about not solving problems.

In this case the placebo is worse than ineffective, it's harmful. It's diverts your interest, enthusiasm and outrage until it's absorbed into this black hole of affectation.
 
Keep calm and carry on "innovating" ... is that the real message of TED? To me that's not inspirational, it's cynical.
 
In the US the rightwing has certain media channels that allow it to bracket reality ... other constituencies have TED. 

Debates about Taiwan’s political status are often framed as a legal or diplomatic puzzle: Is Taiwan part of China, or is it an independent country? But this framing quietly assumes that sovereignty is something clean, continuous, and historically legible. Taiwan’s history resists that assumption. The island’s past is not a single national narrative temporarily interrupted, but a layered sequence of migrations, colonial regimes, authoritarian impositions, and democratic reinvention. As a result, attempts to settle Taiwan’s status by appealing to history alone tend to collapse under their own simplifications.

This essay argues that Taiwan’s history does not merely complicate sovereignty claims — it actively undermines the idea that a single, decisive historical claim is possible at all.

1. A Non-National Origin Story

Unlike many modern states, Taiwan does not begin as a coherent political unit tied to a civilizational core. Prior to the seventeenth century, the island was populated by Austronesian Indigenous societies with no incorporation into Chinese dynastic administration. These societies were oriented toward maritime trade and kinship networks that stretched south and east, not west.

This matters because modern Chinese sovereignty claims often rest on civilizational continuity: the idea that Taiwan has always been part of the Chinese historical world. In reality, Taiwan was peripheral even by imperial standards. When European powers arrived in the seventeenth century, they did not encounter a Chinese province, but a frontier populated by Indigenous groups and scattered settlers.

From the outset, Taiwan’s political meaning was externally imposed rather than internally consolidated.

2. Colonial Rule as the Foundation of the Modern State

If one looks for the origins of Taiwan’s modern institutions, they are not primarily Chinese. They are Japanese.

After 1895, Taiwan became a colony of Japan. Japanese authorities built railways, ports, public health systems, cadastral maps, and a professional bureaucracy. They also imposed cultural assimilation and violently suppressed resistance. This was colonial rule in the full sense — coercive, extractive, and hierarchical.

Yet it was also the first time Taiwan was governed as a unified, legible administrative entity. Ironically, the very features that make Taiwan function today as a modern state — infrastructure, rule-bound administration, statistical governance — were laid down by a foreign empire.

This creates an uncomfortable historical asymmetry: Taiwan’s state capacity owes more to colonial modernity than to any uninterrupted Chinese governance tradition.

3. The Republic of China as a Refugee Regime

When World War II ended, Taiwan was transferred not to a newly formed Taiwanese polity but to the Republic of China (ROC). This transition was neither smooth nor popular. Corruption, mismanagement, and repression culminated in the February 28 Incident, in which thousands of Taiwanese were killed.

Two years later, the ROC lost the Chinese Civil War and fled to Taiwan under Chiang Kai-shek. From that point onward, Taiwan was governed by a state whose primary identity and legitimacy were oriented toward “China,” not Taiwan itself.

This is a crucial point. The ROC did not emerge from Taiwanese self-determination; it arrived as a displaced government, ruling an island it claimed as a temporary base. For decades, Taiwan existed in a paradoxical state: governed locally, imagined nationally elsewhere.

4. Authoritarian Stability and Deferred Self-Definition

During the Cold War, Taiwan’s unresolved status was strategically useful. Backed by the United States, the ROC became an anti-communist stronghold facing the People's Republic of China (PRC). Internally, this period was marked by martial law, political repression, and the systematic silencing of Taiwanese identity.

Yet this same era produced rapid economic development. Land reform, export-led growth, and investment in education transformed Taiwan into a high-income economy. Material success created social stability, but at the cost of political voice.

Crucially, sovereignty remained unresolved not because it was settled, but because it was postponed.

5. Democratization and the Collapse of the Old Narrative

When martial law ended in 1987, Taiwan underwent one of the most rapid and peaceful democratic transitions in East Asia. Competitive elections, press freedom, and civil society followed. With democratization came something more destabilizing for old sovereignty claims: a redefinition of political identity.

A majority of people in Taiwan today identify as Taiwanese rather than Chinese. This shift is not primarily ideological; it is experiential. Citizens vote, pay taxes, serve in the military, and contest power within Taiwan’s institutions — not those of the PRC, and no longer in the name of reclaiming China.

Democracy transformed sovereignty from a historical abstraction into a lived practice.

6. Why History Cannot “Settle” Taiwan

China’s claim over Taiwan often invokes history: dynastic rule, post-war settlement, civil war succession. Taiwan’s counterarguments increasingly invoke democracy: consent, self-governance, and political reality.

The problem is that both sides are partially correct — and therefore incomplete.

History shows that Taiwan has been ruled by many powers, often reluctantly and often externally. Democracy shows that Taiwan today governs itself with popular legitimacy. There is no clean historical moment that confers exclusive sovereignty, because Taiwan’s political identity was never allowed to consolidate until very recently.

This is not a bug in the argument. It is the core fact.

Conclusion

Taiwan’s history does not point toward a single rightful sovereign. Instead, it reveals how sovereignty itself can be contingent, layered, and deferred. The island has moved from Indigenous autonomy to colonial rule, from refugee authoritarianism to democratic self-governance — without ever passing through the classic nation-state origin story that international law prefers.

Asking whether Taiwan “belongs” to China or is “truly independent” misses the deeper reality: Taiwan is the product of historical discontinuity. Its legitimacy today flows less from ancient claims than from contemporary consent.

Any durable resolution of Taiwan’s status will have to reckon with that — not by simplifying history, but by accepting that history has already made simplicity impossible.
/no-think
    """
)