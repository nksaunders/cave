import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x = [val * 0.40212336 for val in x1]

'''
10th mag
flux = 1150000

# m10cdpp = [312.24129386121569, 354.76883029473817, 442.7849919598425, 486.0712606913861, 505.45802945263256]
# m10cdppn = [1.2732594733605547, 1.446678523650349, 1.8055913703884863, 1.9821043839214301, 2.0611598691132098]
# m10cdpp = [309.59794862448155, 360.54927734892209, 471.68796131692596, 639.9102151372615, 803.93757777631492]
# m10cdppn = [1.204689262107806, 1.4029480647812964, 1.8354043513164586, 2.4899808552155127, 3.1282344461745626]
m10cdpp = [18.791998106883419, 26.150987528060533, 29.861926302798992, 34.704987086653688, 63.448982744253385,\
           109.27389687284278, 112.26004069029557, 114.59486391524118, 125.0226093740587, 126.07098170068483,\
           133.71806506097502, 300.23051367522538, 752.7673528684544, 1216.6860738127029, 1392.9562618204204,\
           3606.3967659317377, 7027.8677107829571, 9059.0888336161588, 12057.807983148832, 17213.793205627844]
m10cdppn = [1.1545135400490163, 1.6066236817967632, 1.8346105641580981, 2.1321509969762853, 3.898079877612632,\
            7.3555833076854951, 7.5565904122784922, 7.7137549980696534, 8.4156806420587777, 8.4862500113834933,\
            9.0010001971747471, 20.209497584012336, 50.671231957372335, 81.899118008431458, 93.764440740180575,\
            242.7583586887811, 473.06875568106716, 609.79689124592267, 811.65048256065825, 1158.7167071803556]
'''
m10cdpp = [46.299805620895526, 55.566143004551563, 25.353326169979603, 52.70768977916822, 72.610645032966431, 62.919068643414811, 63.339805719594594, 98.51432302295504, 112.4747797082272, 122.62752550722576, 132.8604636047433, 136.05602700920775, 140.23299049227722, 146.03621700625357, 142.73819513666731, 150.03257334590842, 150.28238824959567, 152.32760255896537, 148.07946603465035, 151.14313352644808]
m10cdppn = [1.7010792507918846, 2.0415293680826698, 0.93149456046197987, 1.936508290653691, 2.6677533522181767, 2.3116797298207112, 2.3271378316578661, 3.6194681284909622, 4.1323826619415671, 4.505399891401523, 4.8813634281570399, 4.9987701111663334, 5.1522339500978331, 5.365447549554399, 5.2442764885579987, 5.5122757868869856, 5.5214541180592693, 5.5965963692725511, 5.4405175953093607, 5.5530783530001164]
nmv10 = m10cdpp[0] / m10cdppn[0]
m10cdppnn = [(val - nmv10)/nmv10 for val in m10cdpp]


'''
11th mag
flux = 454000

# m11cdpp = [445.84291666328073, 451.05992319193638, 537.6077838874528, 541.60906003746652, 540.17527570643813]
# m11cdppn = [1.1717027442592469, 1.1854133598998244, 1.4128664876643089, 1.4233820887208375, 1.419614014871162]
# m11cdpp = [446.52185630132101, 479.20150336396364, 559.48066459617098, 707.2823423448034, 855.73671862785341]
# m11cdppn = [1.0968388372936619, 1.177113308927068, 1.3743114989422611, 1.7373723840570643, 2.1020365615784784]
m11cdpp = [42.052230307811875, 62.398939086588506, 69.899736498361918, 79.536199203339081, 88.729716625453207,\
            206.6327601234544, 267.49847819764102, 271.1514717499723, 285.09384979706084, 284.41885350381244,\
            290.754431753105, 404.04995341987581, 802.19839008791519, 1228.4152401834069, 1562.8515669071087,\
            4112.3248602197464, 7770.403129876504, 8881.0399947695041, 13202.449384509073, 18303.761851202951]
m11cdppn = [1.0938920812071411, 1.623164927113651, 1.8182809252763867, 2.0689513455290851, 2.3080995627075289,\
            5.9034609768918545, 7.6423836494960646, 7.7467490215316559, 8.145080267152558, 8.1257957438545212,\
            8.3068020805960785, 11.543634858790522, 22.918664440140407, 35.095603569875713, 44.650389572292987,\
            117.48838529817375, 221.99902679777165, 253.72972326369526, 377.19161615163654, 522.93520037355063]
'''
m11cdpp = [68.155970862181718, 64.130267982752386, 71.063079435821876, 96.9567198629871, 99.69694824500327, 112.7741982607624, 111.46936177300651, 129.84716752647014, 187.60416885429072, 238.45291240960395, 270.23251668338884, 293.13077111711232, 293.51580718451225, 289.09246526730431, 291.68683758710682, 301.07522925756052, 300.50229404308584, 308.29223371470414, 292.59907853360357, 298.53474920888266]
m11cdppn = [1.1250798386540855, 1.0586258348630424, 1.1730687265476307, 1.6005061531653315, 1.6457402781717552, 1.8616120521579331, 1.8400725566956369, 2.1434428773053007, 3.0968624664187305, 3.9362444819820768, 4.4608440379203227, 4.8388353434194098, 4.8451912989007324, 4.7721732969917374, 4.8149997134310878, 4.9699779208015658, 4.9605202335218426, 5.0891121082761721, 4.8300584659353047, 4.9280411271925351]
nmv11 = m11cdpp[0] / m11cdppn[0]
m11cdppnn = [(val - nmv11)/nmv11 for val in m11cdpp]

'''
12th mag
flux = 159000
# m12cdpp = [637.81563556843287, 617.36846017067558, 680.71054910724308, 685.44862849027743, 660.686867489576]
# m12cdppn = [1.0801966873093451, 1.045567603452211, 1.1528429833910474, 1.1608673361480963, 1.1189311233723378]
# m12cdpp = [673.34870364327287, 691.72025880522733, 750.74171415797002, 867.91401099393863, 1015.2629888184431]
# m12cdppn = [1.0154746194862541, 1.0431806919662567, 1.1321907243485327, 1.308897819647066, 1.5311142527944561]
m12cdpp = [113.21972909949879, 130.38231643012813, 185.66737376238092, 205.80422066023849, 227.16195549528933,\
           255.24536752178437, 589.5132400646504, 708.78791961716377, 783.8909104313426, 794.70459849399663,\
           803.39972483465624, 865.83558630841344, 1145.2113618055544, 1339.4490427525911, 1682.6918726911413,
           4511.2472786818207, 7375.1629596071871, 10118.925851207645, 15299.495545968337, 18663.672108026385]
m12cdppn = [1.063730742554023, 1.2249780084727984, 1.7443964486679475, 1.933587707768474, 2.1342495475030927,\
           2.6461779390194367, 6.111597423940748, 7.3481410242414915, 8.1267482106379223, 8.238855799773269,
           8.3289998510563024, 8.9762844652283071, 11.872626996318068, 13.886326485710279, 17.444791084416355,
           46.768970352774254, 76.459736409203217, 104.90485533201246, 158.61282022449345, 193.4898872910465]
'''
m12cdpp = [126.01809371618796, 128.90351580180487, 150.33633644251813, 163.6532556579759, 214.23512091731945, 253.39766061312812, 269.33962480715587, 289.60211776962115, 301.48458294638442, 444.04430807057861, 653.97089521379598, 773.29558818022747, 779.80180037151661, 764.5691285938658, 770.80111464058905, 798.42335275346636, 793.80620849986747, 831.47445382491856, 772.73386766383214, 796.63698017491993]
m12cdppn = [1.07447984745765, 1.0990820913972148, 1.281826752687568, 1.3953720453140097, 1.8266529293939227, 2.1605681509110473, 2.2964956098198614, 2.469261596872915, 2.5705761699952774, 3.7860964749605941, 5.5760131502512804, 6.5934224294715742, 6.6488969544121579, 6.5190172017086754, 6.5721535666494288, 6.8076716364657059, 6.7683040479948797, 7.0894783277932341, 6.5886329793460261, 6.7924403210323687]
nmv12 = m12cdpp[0] / m12cdppn[0]
m12cdppnn = [(val - nmv12)/nmv12 for val in m12cdpp]

'''
13th mag
flux = 68000

# m13cdpp = [823.51993040031107, 776.46634763751729, 814.92968708599233, 835.45350532180032, 794.41229576850162]
# m13cdppn = [1.0480756418876049, 0.98819158548929242, 1.0371430287919525, 1.0632632393386756, 1.0110309976423417]
# m13cdpp = [927.26098528952934, 929.57057153990957, 981.4924090743782, 1075.975865157247, 1266.0690412668944]
# m13cdppn = [0.99462187862371743, 0.99709924481472501, 1.0527929453040255, 1.1541401539959359, 1.3580426527910199]
m13cdpp = [260.86608932542225, 273.84796289504703, 408.46210618977227, 452.71625252995256, 495.18069092783071,\
           539.08995923685393, 710.35453854364482, 892.22545960832599, 1705.2743259659169, 1819.413596652128,\
           1847.4447817270986, 1874.4514989237887, 2073.8361997369816, 2195.719975757364, 3319.0081148766026,\
           5219.1265306099403, 8848.8705085484125, 10362.133037619524, 14055.664090686023, 18750.779097005445]
m13cdppn = [1.0668135245596371, 1.1199029787466952, 1.6704083703642127, 1.8513859821175425, 2.0250445719052461,\
           2.4029197754901688, 3.1663082181906583, 3.9769729789136914, 7.6010271204083715, 8.1097873115278283,\
           8.2347324858776219, 8.3551112347507477, 9.2438412737825306, 9.7871215384073693, 14.794024814595575,
           23.263542821204652, 39.442630253850794, 46.187791046236512, 62.651200643707973, 83.579033751296507]
'''
m13cdpp = [276.3879051703185, 279.01371684299613, 314.92473043509352, 308.56357517259124, 350.09385358223165, 527.22806756285831, 606.40667989225108, 625.367096697649, 666.34247605684664, 694.9764563578043, 796.76902942068273, 1110.3491548197921, 1411.7332780510449, 1570.7308627801522, 1625.2170084171441, 1702.6643636563992, 1773.6672151363823, 1833.0934705999603, 1673.6693608456942, 1704.3334882580389]
m13cdppn = [1.1417091464983362, 1.1525559062428725, 1.3008978991852131, 1.2746211012152751, 1.44617527500479, 2.1778851236497632, 2.5049578508293666, 2.583279951338084, 2.7525419361083894, 2.8708238022780068, 3.2913107683768734, 4.5866543439491743, 5.8316184095455421, 6.4884090771563496, 6.7134816279716976, 7.0334028408564473, 7.326702957995094, 7.5721822215073811, 6.9136296550821728, 7.0402976969217406]
nmv13 = m13cdpp[0] / m13cdppn[0]
m13cdppnn = [(val - nmv13)/nmv13 for val in m13cdpp]

'''
14th mag
flux = 28000

# m14cdpp = [987.83546813148746, 914.31356230014694, 943.62011726871913, 955.23347913673308, 937.96020160430248]
# m14cdppn = [1.0374214403987652, 0.96020898558310219, 0.9909866296842299, 1.0031829427198167, 0.98504260555213308]
# m14cdpp = [1229.9048325118938, 1213.6498499991944, 1254.7297725032199, 1331.931940062861, 1542.510761142601]
# m14cdppn = [0.99984283012684638, 0.98662845183550607, 1.0200240974920323, 1.0827850783941966, 1.2539737093089434]
m14cdpp = [629.58525041123539, 655.26508649840707, 758.40166781355242, 1052.7973786918615, 1130.5333416346054,\
           1219.3584529731656, 1389.1841767363337, 1408.3925956752876, 1552.9642830044761, 2848.6813682845977,\
           4146.1455740426773, 4149.5290203609929, 4305.2956403766675, 4813.8351775522506, 6521.2238919111169,\
           7995.7637075488501, 10249.234663225308, 11763.36788879043, 15950.88703960838, 19081.903726687848]
m14cdppn = [1.0688067546733779, 1.1124017757621762, 1.2874901767238329, 1.7872670125504055, 1.9192344025424273,\
            2.2149139802885882, 2.5233953533077385, 2.5582866484337869, 2.8208951132691698, 5.1745113773693872,\
            7.5313012132461736, 7.5374471029422843, 7.8203906979891551, 8.7441316436259786, 11.845532321990438,\
            14.523972647872236, 18.617309034504, 21.367669144893497, 28.974123741782233, 34.661485498138376]
'''
m14cdpp = [661.03468122078777, 666.90571280440111, 687.60991624493704, 702.54562340611869, 755.39064779720286, 722.99782218781581, 1253.5351809035255, 1341.4793510648412, 1487.1588791146071, 1468.1044380923436, 1593.8185949832894, 1686.3095119370396, 1867.3215438819007, 2252.0091598887043, 2653.4747237749266, 3075.0635056509482, 3476.7196222737257, 3652.7194365869886, 3573.999977794876, 3548.7196624926873]
m14cdppn = [1.1597387482430321, 1.1700390592752772, 1.2063631246588848, 1.2325667699732945, 1.3252796399319564, 1.2684487109481595, 2.1992391063277545, 2.3535309532090696, 2.609115415423755, 2.5756857419022081, 2.7962423678872184, 2.9585111614907502, 3.2760840110074683, 3.9509913145522808, 4.655334344882001, 5.3949821425788622, 6.0996594842517613, 6.4084387512725414, 6.2703310101880136, 6.2259784791383508]
nmv14 = m14cdpp[0] / m14cdppn[0]
m14cdppnn = [(val - nmv14)/nmv14 for val in m14cdpp]

rawcdpp = [[134.39241328887053, 157.56253752668883, 289.68513955311334, 156.64018747499941, 164.64638764643132, 191.72758206380163, 194.84013752333797, 304.47679688982652, 181.86449915767554, 350.09413641402625, 247.34015499982192, 346.8948602220849, 243.31444898418275, 333.21009380864632, 314.55470063678973, 326.34456521687849, 355.95357887026023, 422.69412233185619, 266.36742991970476, 348.12848077887975],
           [289.55959144161204, 309.57713865720928, 387.52677831087323, 303.73233704529804, 303.19047238079003, 339.26408240587403, 313.79355595778787, 397.20202802989877, 316.85919431069391, 436.5685931797251, 357.43003166946744, 427.18363267404663, 357.01173892502072, 429.17816158729948, 402.93365895173969, 426.30544930692821, 433.06527979421514, 494.33257974280531, 376.98945678010102, 432.76004404243281],
           [798.16592272470461, 839.54146754574731, 839.55592735560037, 803.88317276420571, 813.52702380353946, 878.20269494166541, 813.44661843755046, 854.05749806646566, 825.99298831011117, 803.88386369401599, 821.18381872990062, 890.99318319987026, 847.28463109781399, 879.90612047552213, 843.48648542134708, 883.9253755990419, 880.40047532298627, 925.94984331540593, 828.81639415173606, 852.59472783342517],
           [1621.4959153571838, 1636.459091574141, 1624.1778879268181, 1527.4252375467597, 1605.2859552861851, 1742.8491728852387, 1684.1371985659971, 1609.1874661099121, 1619.0377282946479, 1551.5202877093111, 1584.5614114202992, 1686.7084772432986, 1747.275886044984, 1881.2760827264644, 1865.5962904474661, 1885.667619732337, 1925.4146721216769, 1965.2396822040382, 1836.185396227341, 1898.2536922028546],
           [3345.7740156811892, 3214.8087941211465, 3033.2463984475307, 3043.879049292349, 3239.3394557267679, 3450.0180100926345, 3200.5168378306726, 3278.3488730031445, 3188.7097528068398, 3111.9884748600316, 3103.249637564908, 3142.4295087185069, 3416.2178811316085, 3686.6337109659926, 4124.4385758891513, 4449.897564537906, 4489.0069151096423, 4653.4230769465112, 4410.1530477479382, 4513.4181579600054]]

magcdpps = [m10cdpp,m11cdpp,m12cdpp,m13cdpp,m14cdpp]

for i in range(5):
    fig = pl.figure()
    pl.plot(x,magcdpps[i], label = r'$\mathrm{Detrended}$',color='k')
    pl.plot(x,rawcdpp[i], label = r'$\mathrm{Raw}$',color='r')
    pl.xlim([x[0],x[-1]])
    pl.xlabel("Motion (pixels)")
    pl.ylabel("CDPP")
    pl.title('Mag %i: CDPP vs. Motion' % (i+10))
    legend = pl.legend(loc=0)
    pl.savefig('m%i.png'%(i+10))

pl.show()
'''
Plots


fig1 = pl.figure()
pl.plot(x,m10cdpp,'m', label=r'$\mathrm{K_p\ Mag=10}$')
pl.plot(x,m11cdpp,'k', label=r'$\mathrm{K_p\ Mag=11}$')
pl.plot(x,m12cdpp,'r', label=r'$\mathrm{K_p\ Mag=12}$')
pl.plot(x,m13cdpp,'b', label=r'$\mathrm{K_p\ Mag=13}$')
pl.plot(x,m14cdpp,'g', label=r'$\mathrm{K_p\ Mag=14}$')
pl.title("CDPP vs. Motion")
legend = pl.legend(loc=0)
pl.xlim([x[0],x[-1]])
pl.xlabel("Motion (pixels)")
pl.ylabel("CDPP")
pl.yscale('log')

fig2 = pl.figure()
pl.plot(x,m10cdppn,'m', label=r'$\mathrm{K_p\ Mag=10}$')
pl.plot(x,m11cdppn,'k', label=r'$\mathrm{K_p\ Mag=11}$')
pl.plot(x,m12cdppn,'r', label=r'$\mathrm{K_p\ Mag=12}$')
pl.plot(x,m13cdppn,'b', label=r'$\mathrm{K_p\ Mag=13}$')
pl.plot(x,m14cdppn,'g', label=r'$\mathrm{K_p\ Mag=14}$')
pl.title("Normalized CDPP vs. Motion")
pl.xlabel("Motion (pixels)")
pl.ylabel("Normalized CDPP")
legend = pl.legend(loc=0)
pl.xlim([x[0],x[-1]])

fig3 = pl.figure()
kpmag = [10,11,12,13,14]
cmap = mpl.cm.autumn
for i in range(20):
    cdpp = [m10cdpp[i],m11cdpp[i],m12cdpp[i],m13cdpp[i],m14cdpp[i]]
    pl.plot(kpmag, cdpp, color=cmap(i/20),label=('%.1f pixels' % x[i]))

cdpp1 = [m10cdpp[0],m11cdpp[0],m12cdpp[0],m13cdpp[0],m14cdpp[0]]
cdpp2 = [m10cdpp[1],m11cdpp[1],m12cdpp[1],m13cdpp[1],m14cdpp[1]]
cdpp3 = [m10cdpp[2],m11cdpp[2],m12cdpp[2],m13cdpp[2],m14cdpp[2]]
cdpp4 = [m10cdpp[3],m11cdpp[3],m12cdpp[3],m13cdpp[3],m14cdpp[3]]
cdpp5 = [m10cdpp[4],m11cdpp[4],m12cdpp[4],m13cdpp[4],m14cdpp[4]]
pl.plot(kpmag,cdpp1,'m', label=r'$\mathrm{Motion\ Coefficient=1}$')
pl.plot(kpmag,cdpp2,'k', label=r'$\mathrm{Motion\ Coefficient=2}$')
pl.plot(kpmag,cdpp3,'r', label=r'$\mathrm{Motion\ Coefficient=3}$')
pl.plot(kpmag,cdpp4,'b', label=r'$\mathrm{Motion\ Coefficient=4}$')
pl.plot(kpmag,cdpp5,'g', label=r'$\mathrm{Motion\ Coefficient=5}$')

pl.xlabel('Kp Mag')
pl.ylabel('CDPP')
pl.title('CDDPP vs. Kp Mag')
legend = pl.legend(loc=0)
pl.show()
'''
