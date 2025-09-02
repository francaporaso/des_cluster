SELECT
--metacal.unsheared_coadd_object_id,
--metacal.unsheared_ra, metacal.unsheared_dec,
--metacal.unsheared_e_1, metacal.unsheared_e_2, metacal.unsheared_weight, 
--metacal.unsheared_r11, metacal.unsheared_r12, metacal.unsheared_r21, metacal.unsheared_r22, 
--metacal.unsheared_mask_frac,
--
--metacal.sheared_1p_e_1, metacal.sheared_1p_e_2, metacal.sheared_1p_weight,
--metacal.sheared_1m_e_1, metacal.sheared_1m_e_2, metacal.sheared_1m_weight,
--metacal.sheared_2p_e_1, metacal.sheared_2p_e_2, metacal.sheared_2p_weight,
--metacal.sheared_2m_e_1, metacal.sheared_2m_e_2, metacal.sheared_2m_weight,
--
gold.dnf_zmean_sof, gold.dnf_zsigma_sof
--MAX(gold.dnf_zmean_sof) as zmean_max, MAX(gold.dnf_zsigma_sof) as zsigma_max
--
FROM des_metacal_v03_004 as metacal
INNER JOIN des_y3_gold_v2_2_c as gold ON metacal.unsheared_coadd_object_id = gold.coadd_object_id
WHERE gold.flags_footprint=1 
AND gold.flags_foreground=0 
AND (gold.flags_badregions&110)=0 
AND (gold.flags_gold&1111000)=0
-- Sevilla-Noarbe+21 table 7 'basic object detections for shear studies'
AND gold.dnf_zsigma_sof/gold.dnf_zmean_sof < 0.1
AND gold.dnf_zmean_sof < 2
AND metacal.unsheared_flags=0 
AND metacal.unsheared_snr>10 
AND metacal.unsheared_snr<1000
AND metacal.unsheared_t/metacal.unsheared_mcal_psf_t > 0.5
AND metacal.unsheared_t < 10
