import numpy as np
x=[0.48138371769936095, 0.022058823529411766, 0.01742177001343828, 0.14967105263157895, 0.020849844540632812, 0.5952380952380952, 0.4644293226105784, 0.33463770071515314, 0.5846021874329831, 0.049636285836542574, 0.9144420394420394, 0.034577741311609914, 0.23109046546546547, 0.2574335078775576, 0.5081795172541514, 0.9649849935686723, 0.8460093896713615, 0.024853296513634795, 0.02365208545269583, 0.029850746268656716, 0.03014256619144603, 0.029136316337148804, 0.6849345611023165, 0.7276550998948476, 0.5893115069812229, 0.021076487252124645, 0.8367346938775511, 0.5307831866659884, 0.7631472328012316, 0.020368574199806012, 0.730636033448771, 0.6514806378132119, 0.23834664500699426, 0.5297256097560976, 0.037891836031691356, 0.5300042374262525, 0.02980698753970193, 0.6253773493037297, 0.9758349778456268, 0.09370816599732262, 0.6466876971608833, 0.6204413258240237, 0.36160672141980554, 0.0, 0.02951676099259904, 0.42744872532106576, 0.6216530849825378, 0.6882900071547817, 0.7022900763358778, 0.26271619353426257, 0.4688265722842363, 0.8048684727129957, 0.6157865273371897, 0.9746473113882641, 0.6135656502800249, 0.49755756042452304, 0.36526134573825136, 0.9153498033978782, 0.16756491915727584, 0.03713207547169811, 0.47348687734333156, 0.022789947447838077, 0.5875368309952065, 0.036585365853658534, 0.7732634338138925, 0.6076330766655507, 0.2991985752448798, 0.3531803659599187, 0.0, 0.5979381443298969, 0.02132701421800948, 0.10744945998338411, 0.4766804074540933, 0.5490836932415081, 0.3275080906148867, 0.671429129093961, 0.9053184604256291, 0.8600095556617295, 0.6544223781792843, 0.21643286573146292, 0.014234875444839857, 0.4964039336562454, 0.027902285970416853, 0.9813557414562779, 0.0686633533873921, 0.0, 0.46348231682116076, 0.03629417382999045, 0.4086928316574765, 0.555667001003009, 0.0, 0.5288461538461539, 0.28716216216216217, 0.0, 0.03253388946819604, 0.24106200030170463, 0.7546840721108643, 0.3412594978729683, 0.03492884864165589, 0.9409368635437881, 0.03479799469183132, 0.02681938818270708, 0.6883910386965377, 0.6446755109358193, 0.6890124264224984, 0.012664057103384757, 0.6033903925063208, 0.679474043715847, 0.02126176368072499, 0.027196652719665274, 0.02964254577157803, 0.017197452229299363, 0.882193802262666, 0.7030093981203759, 0.027013871988318325, 0.7260895286925656, 0.5484977044226187, 0.029970029970029972, 0.8077009178419521, 0.02354145342886387, 0.4316057774001699, 0.022375841368018918, 0.41338823433486893, 0.17127213619642398, 0.3475901534211435, 0.4235294117647059, 0.4549155789725781, 0.12583982083822118, 0.7169208656072826, 0.5251298441869756, 0.9173716632443532, 0.6765065038492168, 0.5746962115796997, 0.0219134918361501, 0.6954740937947983, 0.0, 0.023068857042991962, 0.723848613030671, 0.7642458100558659, 0.3998384491114701, 0.5271317829457365, 0.4616168045830681, 0.41818384898916566, 0.03550561797752809, 0.4687889618485579, 0.24528301886792453, 0.47677394989719274, 0.039242685025817556, 0.3094887067287846, 0.693966693966694, 0.7388373229115779, 0.04666666666666667, 0.32639488911580655, 0.08972554539057002, 0.08079056865464633, 0.5265225933202358, 0.24312874560859682, 0.5989195800631943, 0.5036117158123887, 0.5941329730772942, 0.5026353001671165, 0.5219936067714196, 0.35500711912672045, 0.43502202643171806, 0.7027279812938425, 0.021886120996441282, 0.028930600998794558, 0.7797918600674256, 0.02547624512279091, 0.9176521802220127, 0.6450608807104276, 0.588126404865794, 0.02889191972611128, 0.0, 0.02169557106335936, 0.9606986899563319, 0.03453847707533832, 0.8896210873146623, 0.02486069438491213, 0.1587574149235092, 0.023742227247032222, 0.03352752348115563, 0.22725238663484487, 0.7026866557815051, 0.028011842404919152, 0.09975036935146976, 0.02078965758211041, 0.024399690162664602, 0.7426150265560661, 0.5876567716322295, 0.02939068100358423, 0.034572578735496094, 0.029382604776926544, 0.8521441334127456, 0.4601114867952116, 0.48227117446106726, 0.030724213606437453, 0.9386944113290705, 0.7747068676716918, 0.6865671641791045, 0.04310055780778493, 0.48388757335529703]

print('before',np.mean(x))
x=sorted(x)[69:]
#print(x[0])

print('after',np.mean(x))

