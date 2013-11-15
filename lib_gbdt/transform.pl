#!/usr/bin/perl


while(<>)
{
	chomp;
	($label,$qid, @list)=split(" ",$_);

	@list_new=();
	foreach $fv (@list) {
		($f,$v)=split(":",$fv);
		$f=$f-1;
		$fv_n=$f.":".$v;
		push(@list_new, $fv_n);
	}

	print "$label";
	foreach $fv_n (@list_new) {
		print " $fv_n";
	}
	print "\n";
}
